import pyrealsense2 as rs

def get_camera_configuration():
    ## Setup stream 
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    ## Start streaming
    profile = pipeline.start(config)

    ## Create an align object 
    align = rs.align(rs.stream.color)

    return pipeline, align