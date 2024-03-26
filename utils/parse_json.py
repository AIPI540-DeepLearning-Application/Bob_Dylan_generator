import json
import sys

def parse_json_file(file_path):
    with open(file_path, 'r') as f:
        json_str = f.read()

    # remove
    json_str = json_str.strip('[]')
    json_objects = json_str.split('][')

    parsed_data = []

    for json_obj in json_objects:
        # parse json str as dict
        data = json.loads(json_obj)
        parsed_data.append(data)
    # print result
    # for data in parsed_data:
    #     print(data)
    return parsed_data

def save_as_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    # file_path = '/path/to/your/json/file.json'
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    parsed_data = parse_json_file(input_file_path)

    save_as_json(parsed_data, output_file_path)

if __name__ == '__main__':
    main()