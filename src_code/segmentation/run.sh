echo hybrinets
#python tensorrt_inference.py --input 1.mp4 --output output_hybridnets_c0_384x640_simplified_fp32.avi --engine ../models/data_visionin/HybridNets/orinAGX/hybridnets_c0_384x640_simplified_fp32.trt --anchor ../models/data_visionin/HybridNets/anchors/hybridnets_c0_anchors_384x640.npy --dtype FP32 --input_mode large --nc 13

#python tensorrt_inference.py --input 1.mp4 --output output_hybridnets_c0_384x640_simplified_fp16.avi --engine ../models/data_visionin/HybridNets/orinAGX/hybridnets_c0_384x640_simplified_fp16.trt --anchor ../models/data_visionin/HybridNets/anchors/hybridnets_c0_anchors_384x640.npy --dtype FP16 --input_mode large --nc 13

#python tensorrt_inference.py --input 1.mp4 --output output_hybridnets_c0_384x640_simplified_int8.avi --engine ../models/data_visionin/HybridNets/orinAGX/hybridnets_c0_384x640_simplified_int8.trt --anchor ../models/data_visionin/HybridNets/anchors/hybridnets_c0_anchors_384x640.npy --dtype INT8 --input_mode large --nc 13

echo hybrinets_small
#python tensorrt_inference.py --input 1.mp4 --output output_hybridnets_c0_256x512_simplified_fp32.avi --engine ../models/data_visionin/HybridNets/orinAGX/hybridnets_c0_256x512_simplified_fp32.trt --anchor ../models/data_visionin/HybridNets/anchors/hybridnets_c0_anchors_256x512.npy --dtype FP32 --input_mode small --nc 13

#python tensorrt_inference.py --input 1.mp4 --output output_hybridnets_c0_256x512_simplified_fp16.avi --engine ../models/data_visionin/HybridNets/orinAGX/hybridnets_c0_256x512_simplified_fp16.trt --anchor ../models/data_visionin/HybridNets/anchors/hybridnets_c0_anchors_256x512.npy --dtype FP16 --input_mode small --nc 13

#python tensorrt_inference.py --input 1.mp4 --output output_hybridnets_c0_256x512_simplified_int8.avi --engine ../models/data_visionin/HybridNets/orinAGX/hybridnets_c0_256x512_simplified_int8.trt --anchor ../models/data_visionin/HybridNets/anchors/hybridnets_c0_anchors_256x512.npy --dtype INT8 --input_mode small --nc 13


echo xception
#python tensorrt_inference.py --input 1.mp4 --output output_xception_384x640_simplified_fp32.avi --engine ../models/data_visionin/Xception/orinAGX/xception65_384x640_simplified_fp32.trt --anchor ../models/data_visionin/Xception/anchors/xception65_anchors_384x640.npy --dtype FP32 --input_mode large --nc 13

#python tensorrt_inference.py --input 1.mp4 --output output_xception_384x640_simplified_fp16.avi --engine ../models/data_visionin/Xception/orinAGX/xception65_384x640_simplified_fp16.trt --anchor ../models/data_visionin/Xception/anchors/xception65_anchors_384x640.npy --dtype FP16 --input_mode large --nc 13

#python tensorrt_inference.py --input 1.mp4 --output output_xception_384x640_simplified_int8.avi --engine ../models/data_visionin/Xception/orinAGX/xception65_384x640_simplified_int8.trt --anchor ../models/data_visionin/Xception/anchors/xception65_anchors_384x640.npy --dtype INT8 --input_mode large --nc 13

echo xception_small
#python tensorrt_inference.py --input 1.mp4 --output output_xception_256x512_simplified_fp32.avi --engine ../models/data_visionin/Xception/orinAGX/xception65_256x512_simplified_fp32.trt --anchor ../models/data_visionin/Xception/anchors/xception65_anchors_256x512.npy --dtype FP32 --input_mode small --nc 13

python tensorrt_inference.py --input 1.mp4 --output output_xception_256x512_simplified_fp16.avi --engine ../models/data_visionin/Xception/orinAGX/xception65_256x512_simplified_fp16.trt --anchor ../models/data_visionin/Xception/anchors/xception65_anchors_256x512.npy --dtype FP16 --input_mode small --nc 13

python tensorrt_inference.py --input 1.mp4 --output output_xception_256x512_simplified_int8.avi --engine ../models/data_visionin/Xception/orinAGX/xception65_256x512_simplified_int8.trt --anchor ../models/data_visionin/Xception/anchors/xception65_anchors_256x512.npy --dtype INT8 --input_mode small --nc 13

