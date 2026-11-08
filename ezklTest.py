import os

# I spent two hours trying to get this to work on Windows, comment out if on Linux/MACos and run as normal (I haven't tested this method)
os.environ.setdefault("HOME", os.environ["USERPROFILE"])
os.environ.setdefault("EZKL_REPO_PATH", os.path.join(os.environ["USERPROFILE"], ".ezkl"))
os.makedirs(os.environ["EZKL_REPO_PATH"], exist_ok=True)
import asyncio

import ezkl

async def main():
    #Use files we already have
    ezkl.gen_settings("network.onnx", "settings.json")
    ezkl.compile_circuit("network.onnx", "network.ezkl", "settings.json")
    #grab srs or we cant progress (You will get an error here if on windows without the above code)
    await ezkl.get_srs(settings_path="settings.json", srs_path="kzg.srs")
    
    ezkl.setup("network.ezkl", "vk.key", "pk.key", "kzg.srs")
    ezkl.gen_witness("input.json", "network.ezkl", "witness.json", "vk.key", "kzg.srs")
    ezkl.prove("witness.json", "network.ezkl", "pk.key", "proof.json", "kzg.srs")
    #validate model
    ok = ezkl.verify("proof.json", "settings.json", "vk.key", "kzg.srs")
    print("Proof valid:", ok)
asyncio.run(main())
