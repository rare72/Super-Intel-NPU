import openvino as ov

core = ov.Core()
device = "NPU"

print(f"--- Querying NPU 3 (Arrow Lake) Memory Specs ---")

try:
    # List all supported properties to see what we CAN ask for
    supported_props = core.get_property(device, "SUPPORTED_PROPERTIES")
    
    # We will look for anything with 'MEM', 'SIZE', or 'CAPABILITY'
    for prop in supported_props:
        if any(keyword in prop for keyword in ["MEM", "SIZE", "CAPABILITY"]):
            try:
                val = core.get_property(device, prop)
                print(f"{prop}: {val}")
            except:
                continue

except Exception as e:
    print(f"❌ Error: {e}")