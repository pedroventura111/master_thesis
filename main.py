import subprocess

#subprocess.run(["python", "process_data.py"])
#subprocess.run(["python", "market_classification.py"])
#subprocess.run(["python", "trading_strategy.py"])
"""
subprocess.run(["python", "market_classification.py", "20y-200-0-normalization",  "1", "0", "0"])
subprocess.run(["python", "market_classification.py", "20y-200-1-normalization",  "1", "0", "1"])
subprocess.run(["python", "market_classification.py", "20y-200-2-normalization",  "1", "0", "2"])
subprocess.run(["python", "market_classification.py", "20y-200-3-normalization",  "1", "1", "0"])
subprocess.run(["python", "market_classification.py", "20y-200-4-normalization",  "1", "1", "1"])
subprocess.run(["python", "market_classification.py", "20y-200-5-normalization",  "1", "1", "2"])

subprocess.run(["python", "market_classification.py", "20y-200-6-normalization",  "0", "0", "0"])
subprocess.run(["python", "market_classification.py", "20y-200-7-normalization",  "0", "0", "1"])
subprocess.run(["python", "market_classification.py", "20y-200-8-normalization",  "0", "0", "2"])
subprocess.run(["python", "market_classification.py", "20y-200-9-normalization",  "0", "1", "0"])
subprocess.run(["python", "market_classification.py", "20y-200-10-normalization", "0", "1", "1"])
subprocess.run(["python", "market_classification.py", "20y-200-11-normalization", "0", "1", "2"])
"""
#subprocess.run(["python", "market_classification.py", "20y-100-star0", "1", "0", "2", "0", "0", "0", "1", "0", "2", "1", "0", "2"])
#subprocess.run(["python", "market_classification.py", "20y-100-star1", "1", "0", "2", "0", "0", "1", "1", "0", "2", "1", "0", "2"])

#subprocess.run(["python", "market_classification.py", "20y-200-new-hyperopt-star", "0", "1", "0", "0", "1", "1", "0", "0", "1", "0", "0", "2"])
#subprocess.run(["python", "market_classification.py", "20y-200-new-star", "0", "1", "0", "0", "1", "1", "0", "0", "1", "0", "0", "2"])

# {file_var_name, all_features, val_diff, div_feat}

subprocess.run(["python", "market_classification.py", "20y-200-star-exp", "0", "0", "1", "0", "1", "0", "1", "0", "2", "0", "1", "1"])
