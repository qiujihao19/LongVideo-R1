# The two command should be run in two separate terminals
# Start the tool server
host=localhost
port=5000
tool_type=get_caption # separate by comma if you want to start multiple tool servers
workers_per_tool=4 # number of workers for the tool server, meaning how many threads will be used to handle a single tool request with multiple trajectories
tool_config_path="./verl_tool/servers/tool_init_config.example.json"
echo $host $port $tool_type $workers_per_tool
python -m verl_tool.servers.serve --host $host --port $port --tool_type $tool_type --workers_per_tool $workers_per_tool --tool_config_path $tool_config_path 

# Test the tool server
host=localhost
port=5000
tool_type=get_caption # separate by comma if you want to start multiple tool servers
python -m verl_tool.servers.tests.test_get_caption_tool_videoqa python --url=http://localhost:$port/get_observation