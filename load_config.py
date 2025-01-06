import configparser

def load_config(config_file="tracking/utils/itwatcher.cfg"):
    config = configparser.ConfigParser()
    config.read(config_file)

    traffic_light = {}
    if config.has_section("traffic_light"):
        traffic_light = {key: config.get("traffic_light", key)
                         for key in config.options("traffic_light")}

    processed_fields = {}
    if config.has_section("processed_fields"):
        processed_fields = {key: config.getboolean("processed_fields", key)
                            for key in config.options("processed_fields")}

    last_processed_fields = {}
    if config.has_section("last_processed_fields"):
        last_processed_fields = {key: config.get("last_processed_fields", key)
                                 for key in config.options("last_processed_fields")}


    return traffic_light, processed_fields, last_processed_fields
