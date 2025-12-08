import os
import json
from wrapper import run

from Test_Data_Event import event

if __name__ == '__main__':

  os.environ["ASSET_API_URL"] = "https://api.algomart.co.uk"
  os.environ["ASSET_API_KEY"] = "wwLYry06y9cP3Lr2HnWz4ofU17PVtGf2"
    
print(
json.dumps(
    run(event, {}),
    indent=2
)
)