import openvino as ov
core = ov.Core()
print('\n' + '='*25 + '\nSUCCESS! Available Devices:', core.available_devices, '\n' + '='*25)
