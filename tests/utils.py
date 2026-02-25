import os
import json


class ResultAnalyzer:
    @staticmethod
    def calculate_time_steps(log_dir):
        time_list = []
        step_list = []
        files = os.listdir(log_dir)
        for file in files:
            if file.endswith(".log"):
                with open(os.path.join(log_dir, file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if "Total steps executed" in line:
                            step_list.append(int(line.split(":")[-1].strip()))
                        if "Total execution time" in line:
                            time_list.append(float(line.split(":")[-1].strip().split(" ")[0]))

        print(f"Log dir: {log_dir}")
        average_time = round(sum(time_list) / len(time_list), 1)
        average_step = round(sum(step_list) / len(step_list), 2)

        # count prolems solved in one step
        one_step_count = sum([1 for step in step_list if step == 1])
        one_step_rate = round(one_step_count / len(step_list), 1)

        # save the step stats
        step_stats = {
            "average_time": average_time,
            "average_step": average_step,
            "one_step_rate": one_step_rate
        }

        return step_stats

    @staticmethod
    def calculate_tool_usage(result_dir):
        """
        Calculate the usage of tools
        Return a dictionary with the tool name as the key and the ratio of times it is used as the value
        """
        tool_usage = {}
        total_problems = 0
        for filename in os.listdir(result_dir
        ):
            if filename.endswith('.json'):
                file_path = os.path.join(result_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    total_problems += 1

                    if 'step_history' in data:
                        for step in data['step_history'].values():
                            if isinstance(step, dict) and 'Planner' in step:
                                tool_name = step['Planner'].get('tool_to_use')
                                tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

                except json.JSONDecodeError as e:
                    print(f"JSON decode error, cannot parse the file: {filename}, Error message: {e}")
                except Exception as e:
                    print(f"Read or parse file error: {filename}, Error class: {type(e).__name__}, details: {e}")

        # Calculate ratios
        total_tool_usage = sum(tool_usage.values())
        for tool in tool_usage:
            tool_usage[tool] = round(tool_usage[tool] / total_tool_usage, 3)

        # Sort the dictionary by value in descending order
        sorted_tool_usage = dict(sorted(tool_usage.items(), key=lambda item: item[1], reverse=True))

        return sorted_tool_usage
    
    @staticmethod
    def calculate_step_count(result_dir):
        step_counts = []
        for filename in os.listdir(result_dir):
            if filename.endswith('.json') and filename.startswith('output_'):
                file_path = os.path.join(result_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    step_count = data.get('step_count', 0)
                    step_counts.append(step_count)

                except json.JSONDecodeError as e:
                    print(f"JSON decode error, cannot parse the file: {filename}, Error message: {e}")
                except Exception as e:
                    print(f"Read or parse file error: {filename}, Error class: {type(e).__name__}, details: {e}")

        if step_counts:
            average_steps = round(sum(step_counts) / len(step_counts), 2)
            max_steps = max(step_counts)
        else:
            average_steps = 0
            max_steps = 0

        return {
            "average_steps": average_steps,
            "max_steps": max_steps,
            "total_problems": len(step_counts)
        }
