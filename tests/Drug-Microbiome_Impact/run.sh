#!/bin/bash
# Usage: bash Drug-Microbiome_Impact/run.sh
# Description: Run evaluation on Drug-Microbiome_Impact dataset

############ Configuration ############
TASK="Drug-Microbiome_Impact"
THREADS=20
DATA_FILE_NAME="data.json"
ADD_TAG=True

# Model Definitions
# Format: "port:modelname,label,enabled_tools,tool_engines,module_engine"
MODELS=(
    "8000:vllm-Eubiota-8b,Eubiota-8b,\
Python_Coder_Tool|MDIPID_Disease_Search_Tool|MDIPID_Microbe_Search_Tool|MDIPID_Gene_Search_Tool|KEGG_Disease_Search_Tool|KEGG_Drug_Search_Tool|KEGG_Gene_Search_Tool|KEGG_Organism_Search_Tool|Perplexity_Search_Tool|PubMed_Search_Tool|Base_Generator_Tool|Google_Search_Tool|Wikipedia_Search_Tool|URL_Context_Search_Tool,\
gpt-4o|gpt-4o|gpt-4o|gpt-4o|None|None|None|None|None|gpt-4o|gpt-4o|None|gpt-4o|gpt-4o,\
Trainable|gpt-4o|gpt-4o|gpt-4o"
)



############################################

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Set project directory to parent (tests/ folder)
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd $PROJECT_DIR

# Loop through all models
for MODEL_SPEC in "${MODELS[@]}"; do
    # Check if MODEL_SPEC contains a port
    if [[ "$MODEL_SPEC" == *":"* ]]; then
        PORT=$(echo "$MODEL_SPEC" | cut -d':' -f1)
        REST=$(echo "$MODEL_SPEC" | cut -d':' -f2-)
        BASE_URL="http://localhost:${PORT}/v1"
        USE_BASE_URL=true
    else
        PORT=""
        REST="$MODEL_SPEC"
        BASE_URL=""
        USE_BASE_URL=false
    fi

    # Parse configuration
    IFS=',' read -r LLM LABEL ENABLED_TOOLS_RAW TOOL_ENGINE_RAW MODULE_ENGINE_RAW <<< "$REST"
    ENABLED_TOOLS=$(echo "$ENABLED_TOOLS_RAW" | tr '|' ',')
    TOOL_ENGINE=$(echo "$TOOL_ENGINE_RAW" | tr '|' ',')
    MODULE_ENGINE=$(echo "$MODULE_ENGINE_RAW" | tr '|' ',')
    MODULE_ENGINE="${MODULE_ENGINE//Trainable/$LLM}"

    echo "========================================"
    echo "TASK: $TASK"
    echo "MODEL: $LLM"
    echo "LABEL: $LABEL"
    if [ "$USE_BASE_URL" = true ]; then
        echo "BASE_URL: $BASE_URL"
    fi
    echo "========================================"

    DATA_FILE="$TASK/data/$DATA_FILE_NAME"
    LOG_DIR="$TASK/logs/$LABEL"
    OUT_DIR="$TASK/results/$LABEL"
    CACHE_DIR="$TASK/cache"

    mkdir -p "$LOG_DIR"
    mkdir -p "$OUT_DIR"

    # Define indices
    indices=($(jq -r "keys | .[]" "$DATA_FILE"))

    # Skip completed indices
    new_indices=()
    for i in "${indices[@]}"; do
        if [ ! -f "$OUT_DIR/output_$i.json" ]; then
            new_indices+=($i)
        else
            echo "Skipping completed: $OUT_DIR/output_$i.json"
        fi
    done
    indices=("${new_indices[@]}")
    echo "Remaining indices: ${#indices[@]}"

    if [ ${#indices[@]} -eq 0 ]; then
        echo "All subtasks completed for $TASK with $LABEL."
    else
        if [ "$USE_BASE_URL" = true ]; then
            run_task() {
                local i=$1
                python solve.py \
                    --index $i \
                    --task $TASK \
                    --data_file $DATA_FILE \
                    --llm_engine_name $LLM \
                    --module_engine "$MODULE_ENGINE" \
                    --root_cache_dir $CACHE_DIR \
                    --output_json_dir $OUT_DIR \
                    --output_types direct \
                    --enabled_tools "$ENABLED_TOOLS" \
                    --tool_engine "$TOOL_ENGINE" \
                    --max_time 300 \
                    --max_steps 10 \
                    --temperature 0.0 \
                    --base_url "$BASE_URL" \
                    --add_tag $ADD_TAG \
                    2>&1 | tee "$LOG_DIR/$i.log"
            }
        else
            run_task() {
                local i=$1
                python solve.py \
                    --index $i \
                    --task $TASK \
                    --data_file $DATA_FILE \
                    --llm_engine_name $LLM \
                    --module_engine "$MODULE_ENGINE" \
                    --root_cache_dir $CACHE_DIR \
                    --output_json_dir $OUT_DIR \
                    --output_types direct \
                    --enabled_tools "$ENABLED_TOOLS" \
                    --tool_engine "$TOOL_ENGINE" \
                    --max_time 300 \
                    --max_steps 10 \
                    --temperature 0.0 \
                    --add_tag $ADD_TAG \
                    2>&1 | tee "$LOG_DIR/$i.log"
            }
        fi

        export -f run_task
        export TASK DATA_FILE LOG_DIR OUT_DIR CACHE_DIR LLM MODULE_ENGINE ENABLED_TOOLS TOOL_ENGINE BASE_URL ADD_TAG

        echo "Starting parallel execution..."
        parallel -j $THREADS run_task ::: "${indices[@]}"
    fi

    # Calculate Scores
    RESPONSE_TYPE="direct_output"
    python calculate_score_unified.py \
        --task_name $TASK \
        --data_file $DATA_FILE \
        --result_dir $OUT_DIR \
        --response_type $RESPONSE_TYPE \
        --output_file "finalresults_$RESPONSE_TYPE.json" \
        | tee "$OUT_DIR/finalscore_$RESPONSE_TYPE.log"

    echo "========================================"
    echo "Completed: $TASK with $LABEL"
    echo "========================================"
done
