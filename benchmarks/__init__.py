
AVALIABLE_BENCHMARKS = ["lvbench", "videomme", "mlvu", "longvideobench", "videoevalpro"]

def get_benchmark(config):

    if config['name'] not in AVALIABLE_BENCHMARKS:
        raise ValueError(f"Benchmark {config['name']} not found.")
    
    if config['name'] == "lvbench":
        from .lvbench import LVBench
        return LVBench(config)
    elif config['name'] == "videomme":
        from .videomme import VideoMME
        return VideoMME(config)
    elif config['name'] == "longvideobench":
        from .longvideobench import LongVideoBench
        return LongVideoBench(config)
    elif config['name'] == "mlvu":
        from .mlvu import MLVU
        return MLVU(config)
    elif config['name'] == "videoevalpro":
        from .videoevalpro import VideoEvalPro
        return VideoEvalPro(config)
