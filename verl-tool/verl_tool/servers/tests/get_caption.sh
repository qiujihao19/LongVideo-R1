host=localhost
port=5000
tool_type=get_caption # separate by comma if you want to start multiple tool servers
workers_per_tool=4 # number of workers for the tool server, meaning how many threads will be used to handle a single tool request with multiple trajectories
python -m verl_tool.servers.serve --host $host --port $port --tool_type $tool_type --workers_per_tool $workers_per_tool 


# python -m verl_tool.servers.tests.test_get_caption_tool python --url=http://localhost:$port/get_observation