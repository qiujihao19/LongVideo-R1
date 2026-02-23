"""
Tool for longvideo r1
"""
from .base import BaseTool, register_tool
import regex as re
import logging
from .utils.caption_videoqa import get_tool_observation, load_tool_config
logger = logging.getLogger(__name__)

@register_tool
class SearchRetrievalTool(BaseTool):
    tool_type = "get_caption"

    def __init__(self, num_workers=1, config_path=None, **kwargs):
        super().__init__(num_workers)
        self.config_path = kwargs.get("config_path", config_path)
        self.tool_cfg = load_tool_config(self.config_path)
        logger.info(f"get_caption initialized with config_path={self.config_path}")
    
    def extract_and_validate(self, action):
        tags = ['think', 'answer', 'tool']
        results = {}
        for tag in tags:
            pattern = fr'<{tag}>(.*?)</{tag}>'
            match = re.search(pattern, action, re.DOTALL)
            if match:
                results[tag] = match.group(1).strip()

        return results

    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute search query via retrieval service.
        
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string containing search query
            extra_field: Additional parameters
            
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        parsed = self.extract_and_validate(action)
        env = self.load_env(trajectory_id)
        if 'tool' not in  parsed:
            if 'answer' in parsed:
                observation = ""
                done = True
                valid = False
                parsed_action = parsed['answer']
            else:
                observation = ""
                done = False
                valid = False 
                parsed_action = ""
        else:
            try:
                parsed_action = parsed['tool']
                results = get_tool_observation(
                    parsed['tool'],
                    extra_field['video_uid'],
                    extra_field['data_source'],
                    extra_field['width'],
                    extra_field['fps'],
                    tool_cfg=self.tool_cfg,
                )
                if results is None:
                    observation = ""
                    done = False
                    valid = False

                else:
                    observation = results
                    done = False
                    valid = True
            except Exception as e:
                logger.error(f"Require caption error for trajectory {trajectory_id}: {e}")
                observation = ""
                done = False
                valid = False  
                parsed_action = ""            

        
        self.update_env(trajectory_id, env, parsed_action, valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
    
