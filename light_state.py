from tracking.utils.load_config import load_config

traffic_light, _, _ = load_config()

_current_light_state = traffic_light.get("current_light_state")

def get_current_light_state():
    return _current_light_state

def set_current_light_state(state):
    global _current_light_state
    _current_light_state = state
