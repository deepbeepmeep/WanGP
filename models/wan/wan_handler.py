# Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models.help_shared import merge_help
from .infos import wan_help

import re
import torch
import numpy as np
import gradio as gr
import cv2
from PIL import Image
from shared.utils.hf import build_hf_url
from shared.utils import files_locator as fl
from .bernini_prompt_infos import get_bernini_infos, get_bernini_prompt_infos
from .shotplan_prompt_infos import SHOTPLAN_PROMPT_ENHANCER, SHOTPLAN_PROMPT_INFOS
from .vace_infos import VACE_INFOS
from .camera_control import infos as camera_infos
from .kiwi.variant_config import get_kiwi_variant_model_def
from .scail2 import (
    SCAIL2_ANIMATE_PREPROCESSING_POSE,
    SCAIL2_ANIMATE_PREPROCESSING_RAW,
    SCAIL2_INJECT_REF_FRAMES_NO,
    SCAIL2_INJECT_REF_FRAMES_SETTING,
    SCAIL2_INJECT_REF_FRAMES_YES,
    SCAIL2_INFOS,
    custom_image_ref_postprocessor_scail2,
    custom_preprocess_scail2,
    preprocess_all_scail2,
    test_scail2,
    test_scail2_replace,
)

def test_vace(base_model_type):
    return base_model_type in ["vace_14B", "vace_14B_2_2", "vace_1.3B", "vace_multitalk_14B", "vace_standin_14B", "vace_lynx_14B", "vace_ditto_14B"]     

def test_class_i2v(base_model_type):
    return base_model_type in ["i2v", "i2v_2_2", "fun_inp_1.3B", "fun_inp", "flf2v_720p",  "fantasy",  "multitalk", "infinitetalk", "i2v_2_2_multitalk", "animate", "animate2", "chrono_edit", "steadydancer", "wanmove", "scail", "scail2_14B", "scail2_1.3B", "i2v_2_2_svi2pro", "fun_camera_1.3B", "uni3c" ]

def test_camera_control(base_model_type):
    return base_model_type in ["fun_camera_1.3B", "uni3c"]

def test_class_t2v(base_model_type):    
    return base_model_type in ["t2v", "t2v_2_2", "shotplan_t2v", "shotplan_t2v_2_2", "alpha", "alpha2", "lynx", "vista4d", "bernini", "bernini_1.3B"]

def test_shotplan(base_model_type):
    return base_model_type in ["shotplan_t2v", "shotplan_t2v_2_2"]

def test_oneframe_overlap(base_model_type):
    return test_class_i2v(base_model_type) and not (test_multitalk(base_model_type) or base_model_type in ["animate", "scail"] or test_scail2(base_model_type) or test_svi2pro(base_model_type))  or test_wan_5B(base_model_type)

def test_class_1_3B(base_model_type):    
    return base_model_type in [ "vace_1.3B", "t2v_1.3B", "recam_1.3B","phantom_1.3B","fun_inp_1.3B", "scail2_1.3B", "bernini_1.3B", "fun_camera_1.3B"]

def test_multitalk(base_model_type):
    return base_model_type in ["multitalk", "vace_multitalk_14B", "i2v_2_2_multitalk", "infinitetalk"]

def test_standin(base_model_type):
    return base_model_type in ["standin", "vace_standin_14B"]

def test_lynx(base_model_type):
    return base_model_type in ["lynx_lite", "vace_lynx_lite_14B", "lynx", "vace_lynx_14B", "alpha_lynx"]

def test_alpha(base_model_type):
    return base_model_type in ["alpha", "alpha2", "alpha_lynx"]

def test_wan_5B(base_model_type):
    return base_model_type in ["ti2v_2_2", "lucy_edit", "kiwi_edit"]

def test_i2v_2_2(base_model_type):
    return base_model_type in ["i2v_2_2", "i2v_2_2_multitalk", "i2v_2_2_svi2pro"]


def test_svi2pro(base_model_type):
    return base_model_type in ["i2v_2_2_svi2pro"]

def test_bernini(base_model_type):
    return base_model_type in ["bernini", "bernini_1.3B"]

class family_handler():
    @staticmethod
    def query_supported_types():
        return ["multitalk", "infinitetalk", "fantasy", "vace_14B", "vace_14B_2_2", "vace_multitalk_14B", "vace_standin_14B", "vace_lynx_14B",
                    "t2v_1.3B", "standin", "lynx_lite", "lynx", "t2v", "t2v_2_2", "vace_1.3B", "vace_ditto_14B", "phantom_1.3B", "phantom_14B",
                    "recam_1.3B", "animate", "animate2", "alpha", "alpha2", "alpha_lynx", "chrono_edit", "shotplan_t2v", "shotplan_t2v_2_2",
                    "i2v", "i2v_2_2", "i2v_2_2_multitalk", "ti2v_2_2", "lucy_edit", "kiwi_edit", "flf2v_720p", "fun_inp_1.3B", "fun_inp", "mocha", "steadydancer", "wanmove", "scail", "scail2_14B", "scail2_1.3B", "vista4d", "i2v_2_2_svi2pro", "bernini", "bernini_1.3B", "fun_camera_1.3B", "uni3c"]


    @staticmethod
    def query_family_maps():

        models_eqv_map = {
            "flf2v_720p" : "i2v",
            "i2v_2_2_svi2pro": "i2v_2_2",
            "t2v_1.3B" : "t2v", 
            "t2v_2_2" : "t2v", 
            "alpha" : "t2v", 
            "alpha2" : "t2v", 
            "lynx" : "t2v", 
            "vista4d": "t2v",
            "standin" : "t2v", 
            "vace_standin_14B" : "vace_14B",
            "vace_lynx_14B" : "vace_14B",
            "vace_14B_2_2": "vace_14B",
            "kiwi_edit": "ti2v_2_2",
        }

        models_comp_map = { 
                    "vace_14B" : [ "vace_multitalk_14B", "vace_standin_14B", "vace_lynx_lite_14B", "vace_lynx_14B", "vace_14B_2_2"],
                    "t2v" : [ "vace_14B", "vace_1.3B" "vace_multitalk_14B", "vace_standin_14B", "vace_lynx_lite_14B", "vace_lynx_14B", "vace_14B_2_2", "t2v_1.3B", "phantom_1.3B","phantom_14B", "standin", "lynx_lite", "lynx", "alpha", "alpha2"],
                    "i2v" : [ "fantasy", "multitalk", "flf2v_720p" ],
                    "i2v_2_2" : ["i2v_2_2_multitalk", "i2v_2_2_svi2pro"],
                    "fantasy": ["multitalk"],
                    }
        return models_eqv_map, models_comp_map

    @staticmethod
    def query_model_family():
        return "wan"
    
    @staticmethod
    def query_family_infos():
        return {"wan":(0, "Wan2.1"), "wan2_2":(1, "Wan2.2") }

    @staticmethod
    def get_lora_dir(base_model_type):
        if test_shotplan(base_model_type):
            return "wan_shotplan_2_2" if base_model_type == "shotplan_t2v_2_2" else "wan_shotplan"
        i2v = test_class_i2v(base_model_type) and not test_i2v_2_2(base_model_type)

        if base_model_type == "animate2":
            return "wan_animate2"
        if base_model_type == "fun_camera_1.3B":
            return "wan_1.3B"
        if test_scail2(base_model_type):
            return "wan_i2v"
        if i2v:
            return "wan_i2v"
        if "1.3B" in base_model_type:
            return "wan_1.3B"
        if test_wan_5B(base_model_type) or base_model_type in ["ovi"]:
            return "wan_5B"
        return "wan"

    @staticmethod
    def set_cache_parameters(cache_type, base_model_type, model_def, inputs, skip_steps_cache):
        i2v =  test_class_i2v(base_model_type)

        resolution = inputs["resolution"]
        width, height = resolution.split("x")
        pixels = int(width) * int(height)

        if cache_type == "mag":
            skip_steps_cache.update({     
            "magcache_thresh" : 0,
            "magcache_K" : 2,
            })
            if base_model_type in ["t2v", "mocha"] and "URLs2" in model_def:
                def_mag_ratios = [1.00124, 1.00155, 0.99822, 0.99851, 0.99696, 0.99687, 0.99703, 0.99732, 0.9966, 0.99679, 0.99602, 0.99658, 0.99578, 0.99664, 0.99484, 0.9949, 0.99633, 0.996, 0.99659, 0.99683, 0.99534, 0.99549, 0.99584, 0.99577, 0.99681, 0.99694, 0.99563, 0.99554, 0.9944, 0.99473, 0.99594, 0.9964, 0.99466, 0.99461, 0.99453, 0.99481, 0.99389, 0.99365, 0.99391, 0.99406, 0.99354, 0.99361, 0.99283, 0.99278, 0.99268, 0.99263, 0.99057, 0.99091, 0.99125, 0.99126, 0.65523, 0.65252, 0.98808, 0.98852, 0.98765, 0.98736, 0.9851, 0.98535, 0.98311, 0.98339, 0.9805, 0.9806, 0.97776, 0.97771, 0.97278, 0.97286, 0.96731, 0.96728, 0.95857, 0.95855, 0.94385, 0.94385, 0.92118, 0.921, 0.88108, 0.88076, 0.80263, 0.80181]
            elif base_model_type in ["i2v_2_2"]:
                def_mag_ratios = [0.99191, 0.99144, 0.99356, 0.99337, 0.99326, 0.99285, 0.99251, 0.99264, 0.99393, 0.99366, 0.9943, 0.9943, 0.99276, 0.99288, 0.99389, 0.99393, 0.99274, 0.99289, 0.99316, 0.9931, 0.99379, 0.99377, 0.99268, 0.99271, 0.99222, 0.99227, 0.99175, 0.9916, 0.91076, 0.91046, 0.98931, 0.98933, 0.99087, 0.99088, 0.98852, 0.98855, 0.98895, 0.98896, 0.98806, 0.98808, 0.9871, 0.98711, 0.98613, 0.98618, 0.98434, 0.98435, 0.983, 0.98307, 0.98185, 0.98187, 0.98131, 0.98131, 0.9783, 0.97835, 0.97619, 0.9762, 0.97264, 0.9727, 0.97088, 0.97098, 0.96568, 0.9658, 0.96045, 0.96055, 0.95322, 0.95335, 0.94579, 0.94594, 0.93297, 0.93311, 0.91699, 0.9172, 0.89174, 0.89202, 0.8541, 0.85446, 0.79823, 0.79902]
            elif test_wan_5B(base_model_type):
                if inputs.get("image_start", None) is not None and inputs.get("video_source", None) is not None : # t2v
                    def_mag_ratios = [0.99505, 0.99389, 0.99441, 0.9957, 0.99558, 0.99551, 0.99499, 0.9945, 0.99534, 0.99548, 0.99468, 0.9946, 0.99463, 0.99458, 0.9946, 0.99453, 0.99408, 0.99404, 0.9945, 0.99441, 0.99409, 0.99398, 0.99403, 0.99397, 0.99382, 0.99377, 0.99349, 0.99343, 0.99377, 0.99378, 0.9933, 0.99328, 0.99303, 0.99301, 0.99217, 0.99216, 0.992, 0.99201, 0.99201, 0.99202, 0.99133, 0.99132, 0.99112, 0.9911, 0.99155, 0.99155, 0.98958, 0.98957, 0.98959, 0.98958, 0.98838, 0.98835, 0.98826, 0.98825, 0.9883, 0.98828, 0.98711, 0.98709, 0.98562, 0.98561, 0.98511, 0.9851, 0.98414, 0.98412, 0.98284, 0.98282, 0.98104, 0.98101, 0.97981, 0.97979, 0.97849, 0.97849, 0.97557, 0.97554, 0.97398, 0.97395, 0.97171, 0.97166, 0.96917, 0.96913, 0.96511, 0.96507, 0.96263, 0.96257, 0.95839, 0.95835, 0.95483, 0.95475, 0.94942, 0.94936, 0.9468, 0.94678, 0.94583, 0.94594, 0.94843, 0.94872, 0.96949, 0.97015]
                else: # i2v
                    def_mag_ratios = [0.99512, 0.99559, 0.99559, 0.99561, 0.99595, 0.99577, 0.99512, 0.99512, 0.99546, 0.99534, 0.99543, 0.99531, 0.99496, 0.99491, 0.99504, 0.99499, 0.99444, 0.99449, 0.99481, 0.99481, 0.99435, 0.99435, 0.9943, 0.99431, 0.99411, 0.99406, 0.99373, 0.99376, 0.99413, 0.99405, 0.99363, 0.99359, 0.99335, 0.99331, 0.99244, 0.99243, 0.99229, 0.99229, 0.99239, 0.99236, 0.99163, 0.9916, 0.99149, 0.99151, 0.99191, 0.99192, 0.9898, 0.98981, 0.9899, 0.98987, 0.98849, 0.98849, 0.98846, 0.98846, 0.98861, 0.98861, 0.9874, 0.98738, 0.98588, 0.98589, 0.98539, 0.98534, 0.98444, 0.98439, 0.9831, 0.98309, 0.98119, 0.98118, 0.98001, 0.98, 0.97862, 0.97859, 0.97555, 0.97558, 0.97392, 0.97388, 0.97152, 0.97145, 0.96871, 0.9687, 0.96435, 0.96434, 0.96129, 0.96127, 0.95639, 0.95638, 0.95176, 0.95175, 0.94446, 0.94452, 0.93972, 0.93974, 0.93575, 0.9359, 0.93537, 0.93552, 0.96655, 0.96616]
            elif test_class_1_3B(base_model_type): #text 1.3B
                def_mag_ratios = [1.0124, 1.02213, 1.00166, 1.0041, 0.99791, 1.00061, 0.99682, 0.99762, 0.99634, 0.99685, 0.99567, 0.99586, 0.99416, 0.99422, 0.99578, 0.99575, 0.9957, 0.99563, 0.99511, 0.99506, 0.99535, 0.99531, 0.99552, 0.99549, 0.99541, 0.99539, 0.9954, 0.99536, 0.99489, 0.99485, 0.99518, 0.99514, 0.99484, 0.99478, 0.99481, 0.99479, 0.99415, 0.99413, 0.99419, 0.99416, 0.99396, 0.99393, 0.99388, 0.99386, 0.99349, 0.99349, 0.99309, 0.99304, 0.9927, 0.9927, 0.99228, 0.99226, 0.99171, 0.9917, 0.99137, 0.99135, 0.99068, 0.99063, 0.99005, 0.99003, 0.98944, 0.98942, 0.98849, 0.98849, 0.98758, 0.98757, 0.98644, 0.98643, 0.98504, 0.98503, 0.9836, 0.98359, 0.98202, 0.98201, 0.97977, 0.97978, 0.97717, 0.97718, 0.9741, 0.97411, 0.97003, 0.97002, 0.96538, 0.96541, 0.9593, 0.95933, 0.95086, 0.95089, 0.94013, 0.94019, 0.92402, 0.92414, 0.90241, 0.9026, 0.86821, 0.86868, 0.81838, 0.81939]#**(0.5)# In our papaer, we utilize the sqrt to smooth the ratio, which has little impact on the performance and can be deleted.
            elif i2v:
                if pixels >= 1280*720:
                    def_mag_ratios = [0.99428, 0.99498, 0.98588, 0.98621, 0.98273, 0.98281, 0.99018, 0.99023, 0.98911, 0.98917, 0.98646, 0.98652, 0.99454, 0.99456, 0.9891, 0.98909, 0.99124, 0.99127, 0.99102, 0.99103, 0.99215, 0.99212, 0.99515, 0.99515, 0.99576, 0.99572, 0.99068, 0.99072, 0.99097, 0.99097, 0.99166, 0.99169, 0.99041, 0.99042, 0.99201, 0.99198, 0.99101, 0.99101, 0.98599, 0.98603, 0.98845, 0.98844, 0.98848, 0.98851, 0.98862, 0.98857, 0.98718, 0.98719, 0.98497, 0.98497, 0.98264, 0.98263, 0.98389, 0.98393, 0.97938, 0.9794, 0.97535, 0.97536, 0.97498, 0.97499, 0.973, 0.97301, 0.96827, 0.96828, 0.96261, 0.96263, 0.95335, 0.9534, 0.94649, 0.94655, 0.93397, 0.93414, 0.91636, 0.9165, 0.89088, 0.89109, 0.8679, 0.86768]
                else:
                    def_mag_ratios =  [0.98783, 0.98993, 0.97559, 0.97593, 0.98311, 0.98319, 0.98202, 0.98225, 0.9888, 0.98878, 0.98762, 0.98759, 0.98957, 0.98971, 0.99052, 0.99043, 0.99383, 0.99384, 0.98857, 0.9886, 0.99065, 0.99068, 0.98845, 0.98847, 0.99057, 0.99057, 0.98957, 0.98961, 0.98601, 0.9861, 0.98823, 0.98823, 0.98756, 0.98759, 0.98808, 0.98814, 0.98721, 0.98724, 0.98571, 0.98572, 0.98543, 0.98544, 0.98157, 0.98165, 0.98411, 0.98413, 0.97952, 0.97953, 0.98149, 0.9815, 0.9774, 0.97742, 0.97825, 0.97826, 0.97355, 0.97361, 0.97085, 0.97087, 0.97056, 0.97055, 0.96588, 0.96587, 0.96113, 0.96124, 0.9567, 0.95681, 0.94961, 0.94969, 0.93973, 0.93988, 0.93217, 0.93224, 0.91878, 0.91896, 0.90955, 0.90954, 0.92617, 0.92616]
            else: # text 14B
                def_mag_ratios = [1.02504, 1.03017, 1.00025, 1.00251, 0.9985, 0.99962, 0.99779, 0.99771, 0.9966, 0.99658, 0.99482, 0.99476, 0.99467, 0.99451, 0.99664, 0.99656, 0.99434, 0.99431, 0.99533, 0.99545, 0.99468, 0.99465, 0.99438, 0.99434, 0.99516, 0.99517, 0.99384, 0.9938, 0.99404, 0.99401, 0.99517, 0.99516, 0.99409, 0.99408, 0.99428, 0.99426, 0.99347, 0.99343, 0.99418, 0.99416, 0.99271, 0.99269, 0.99313, 0.99311, 0.99215, 0.99215, 0.99218, 0.99215, 0.99216, 0.99217, 0.99163, 0.99161, 0.99138, 0.99135, 0.98982, 0.9898, 0.98996, 0.98995, 0.9887, 0.98866, 0.98772, 0.9877, 0.98767, 0.98765, 0.98573, 0.9857, 0.98501, 0.98498, 0.9838, 0.98376, 0.98177, 0.98173, 0.98037, 0.98035, 0.97678, 0.97677, 0.97546, 0.97543, 0.97184, 0.97183, 0.96711, 0.96708, 0.96349, 0.96345, 0.95629, 0.95625, 0.94926, 0.94929, 0.93964, 0.93961, 0.92511, 0.92504, 0.90693, 0.90678, 0.8796, 0.87945, 0.86111, 0.86189]
            skip_steps_cache.def_mag_ratios = def_mag_ratios
        else:
            if i2v:
                if pixels >= 1280*720:
                    coefficients= [-114.36346466,   65.26524496,  -18.82220707,    4.91518089,   -0.23412683]
                else:
                    coefficients= [-3.02331670e+02,  2.23948934e+02, -5.25463970e+01,  5.87348440e+00, -2.01973289e-01]
            else:
                if test_class_1_3B(base_model_type):
                    coefficients= [2.39676752e+03, -1.31110545e+03,  2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
                else: 
                    coefficients= [-5784.54975374,  5449.50911966, -1811.16591783,   256.27178429, -13.02252404]
            skip_steps_cache.coefficients = coefficients

    @staticmethod
    def query_model_def(base_model_type, model_def):
        extra_model_def = {}
        extra_model_def["device_explicit"] = base_model_type in ["t2v", "t2v_1.3B", "t2v_2_2", "i2v", "i2v_2_2", "flf2v_720p", "fun_inp", "fun_inp_1.3B", "ti2v_2_2", "i2v_2_2_svi2pro", "bernini_1.3B", "kiwi_edit", "vace_14B_2_2", "bernini", "steadydancer", "scail", "scail2_14B", "wanmove",
                                                                 "lynx", "standin", "vace_14B", "vace_ditto_14B", "vace_lynx_14B", "vace_standin_14B", "infinitetalk", "fun_camera_1.3B", "uni3c"]
        if base_model_type not in ["mocha", "recam_1.3B", "scail2_14B", "scail2_1.3B", "vista4d"]:
            extra_model_def["riflex"] = True
        override_text_encoder_urls = model_def.get("text_encoder_URLs", None)
        override_text_encoder_folder = model_def.get("text_encoder_folder", None)
        if "URLs2" in model_def:
            extra_model_def["no_steps_skipping"] = True
            extra_model_def["compile"] = ["transformer","transformer2"]
        text_encoder_folder = "umt5-xxl"
        text_encoder_urls = [
            build_hf_url("DeepBeepMeep/Wan2.1", text_encoder_folder, "models_t5_umt5-xxl-enc-bf16.safetensors"),
            build_hf_url("DeepBeepMeep/Wan2.1", text_encoder_folder, "models_t5_umt5-xxl-enc-quanto_int8.safetensors"),
        ]
        extra_model_def["i2v_class"] = i2v =  test_class_i2v(base_model_type)
        extra_model_def["t2v_class"] = t2v =  test_class_t2v(base_model_type)
        extra_model_def["multitalk_class"] = multitalk = test_multitalk(base_model_type)
        extra_model_def["standin_class"] = standin = test_standin(base_model_type)
        extra_model_def["lynx_class"] = lynx = test_lynx(base_model_type)
        extra_model_def["alpha_class"] = alpha = test_alpha(base_model_type)
        if alpha:
            extra_model_def["specialities"] = [{"name": "alpha output", "aliases": ["RGBA", "transparent video", "transparent background"], "description": "Generate foreground RGB and alpha together. Video saves a checkerboard preview plus RGBA PNG ZIP or ProRes 4444; image mode saves RGBA."}]
        extra_model_def["wan_5B_class"] = wan_5B = test_wan_5B(base_model_type)
        extra_model_def["tiny_vae_architecture"] = "ti2v_2_2" if wan_5B else "t2v"
        extra_model_def["vace_class"] = vace_class = test_vace(base_model_type)
        # control videos and masks in uint8 (VACE / Animate 2: WanGP's preprocessing, SCAIL-2 / SteadyDancer: their own preprocessors)
        extra_model_def["uint8_guides"] = vace_class or base_model_type in ["animate2", "steadydancer"] or test_scail2(base_model_type)
        extra_model_def["bernini_class"] = bernini = test_bernini(base_model_type)
        extra_model_def["shotplan"] = shotplan = test_shotplan(base_model_type)
        extra_model_def["scail2"] = scail2 = test_scail2(base_model_type)
        animate2 = base_model_type == "animate2"
        extra_model_def["animate2"] = animate2
        if animate2:
            extra_model_def.update({
                "no_steps_skipping": True,
                "custom_settings": [{
                    "id": "animate2_kv_cache",
                    "name": "KV cache",
                    "label": "Animate 2 KV Cache",
                    "type": "dropdown",
                    "default": "Disabled",
                    "choices": [("Disabled (Slower but lower VRAM/RAM)", "Disabled"), ("Enabled (Faster, best quality but requires more VRAM/RAM)", "Enabled"),
                                ("INT8 in VRAM (Fastest, about half the VRAM of Enabled)", "INT8"), ("INT8 in RAM (Faster, lowest VRAM)", "RAM")],
                    "info": "Caches the driving-video branch, which is the same at every step. Enabled keeps it exactly on the GPU. INT8 caches its keys and values in about half the memory, so cached steps skip recomputing them; images differ very slightly from Enabled. INT8 in RAM keeps them in system RAM and moves each layer to the GPU as it runs, for the lowest VRAM.",
                }],
            })
        if shotplan:
            extra_model_def.update({
                "prompt_infos": SHOTPLAN_PROMPT_INFOS,
                "prompt_enhancer_def": {
                    "selection": ["T"],
                    "labels": {"TV": "An Enhanced ShotPlan Prompt Relay using existing Text Prompt"},
                    "default": "",
                },
                "text_prompt_enhancer_instructions": SHOTPLAN_PROMPT_ENHANCER,
                "video_prompt_enhancer_instructions": SHOTPLAN_PROMPT_ENHANCER,
                "text_prompt_enhancer_max_tokens": 512,
                "video_prompt_enhancer_max_tokens": 512,
            })
            if base_model_type == "shotplan_t2v_2_2":
                extra_model_def["config_file2"] = "models/wan/configs/t2v_2_2.json"
                extra_model_def["save_quantized_submodel2"] = False
        if vace_class:
            extra_model_def["infos"] = model_def.get("infos", VACE_INFOS)
        if bernini:
            extra_model_def["t2v_class"] = t2v = False
            extra_model_def["prompt_infos"] = get_bernini_prompt_infos(base_model_type)
            bernini_infos = get_bernini_infos(base_model_type)
            if bernini_infos is not None:
                extra_model_def["infos"] = bernini_infos
        extra_model_def["color_correction"] = True
        extra_model_def["svi2pro"] = svi2pro = test_svi2pro(base_model_type)
        extra_model_def["i2v_2_2"] = i2v_2_2 = test_i2v_2_2(base_model_type)
        if base_model_type == "kiwi_edit":
            extra_model_def.update(get_kiwi_variant_model_def(model_def))
            text_encoder_folder = override_text_encoder_folder or extra_model_def["kiwi_text_encoder_folder"]
            kiwi_mllm_repo_id = extra_model_def["kiwi_mllm_repo_id"]
            if override_text_encoder_urls is None:
                text_encoder_urls = [
                    build_hf_url(kiwi_mllm_repo_id, text_encoder_folder, extra_model_def["kiwi_text_encoder_file"]),
                    build_hf_url(kiwi_mllm_repo_id, text_encoder_folder, extra_model_def["kiwi_text_encoder_quanto_file"]),
                ]
            else:
                text_encoder_urls = override_text_encoder_urls
        elif override_text_encoder_urls is not None or override_text_encoder_folder is not None:
            text_encoder_folder = override_text_encoder_folder or text_encoder_folder
            text_encoder_urls = override_text_encoder_urls or text_encoder_urls
        extra_model_def["text_encoder_URLs"] = text_encoder_urls
        extra_model_def["text_encoder_folder"] = text_encoder_folder

        
        if multitalk or base_model_type in ["fantasy"]:
            if multitalk:
                extra_model_def["audio_prompt_choices"] = True                
            extra_model_def["any_audio_prompt"] = True
            extra_model_def["returns_audio"] = True

        if base_model_type in ["vace_multitalk_14B", "vace_standin_14B", "vace_lynx_14B"]:
            extra_model_def["parent_model_type"] = "vace_14B"

        if base_model_type in ["alpha2"]:
            extra_model_def["parent_model_type"] = "alpha"

        group = "wan"
        if test_camera_control(base_model_type):
            profiles_dir = base_model_type
        elif bernini:
            profiles_dir = ["wan_bernini_1.3B", "wan_1.3B"] if base_model_type == "bernini_1.3B" else ["wan_bernini", "wan_2_2"]
            group = "wan2_2"
        elif base_model_type == "animate2":
            profiles_dir = "wan_i2v"
            group = "wan2_2"
        elif base_model_type in ["t2v_2_2", "vace_14B_2_2", "shotplan_t2v_2_2"] or test_i2v_2_2(base_model_type):
            profiles_dir = ["wan_shotplan_2_2", "wan_2_2"] if shotplan else "wan_2_2"
            group = "wan2_2"
        elif base_model_type == "shotplan_t2v":
            profiles_dir = ["wan_shotplan", "wan"]
        elif scail2 or i2v:
            profiles_dir = "wan_i2v"
            if base_model_type in ["chrono_edit"]:
                profiles_dir = "wan_chrono_edit"
        elif test_wan_5B(base_model_type):
            profiles_dir = "wan_2_2_5B"
            group = "wan2_2"
        elif test_class_1_3B(base_model_type):
            profiles_dir = "wan_1.3B"
        elif test_alpha(base_model_type):
            profiles_dir = "wan_alpha"
        else:
            profiles_dir = "wan"

        if  (test_class_t2v(base_model_type) or vace_class or base_model_type in ["chrono_edit"]) and not test_alpha(base_model_type):
            extra_model_def["vae_upsampler"] = [1,2]
            if test_class_t2v(base_model_type) and not wan_5B:
                extra_model_def["vae_upsamplers"] = {"qwen_vae_pid(1.5)": [1]}
                extra_model_def["excluded_spatial_upsamplers"] = ["qwen_pid(1.5)"]
        extra_model_def["vae_block_size"] = 32 if test_wan_5B(base_model_type) or base_model_type in ["scail"] or scail2 else 16

        extra_model_def["profiles_dir"] = profiles_dir if isinstance(profiles_dir, list) else [profiles_dir]
        extra_model_def["group"] = group

        if base_model_type in ["animate"]:
            fps = 30
        elif multitalk:
            fps = 25
        elif base_model_type in ["fantasy"]:
            fps = 23
        elif wan_5B:
            fps = 24
        else:
            fps = 16
        extra_model_def["fps"] =fps
        multiple_submodels = "URLs2" in model_def
        if vace_class: 
            frames_minimum, frames_steps =  17, 4
        else:
            frames_minimum, frames_steps = 5, 4
        extra_model_def.update({
        "frames_minimum" : frames_minimum,
        "frames_steps" : frames_steps, 
        "sliding_window" : not shotplan and (base_model_type in ["multitalk", "infinitetalk", "t2v", "t2v_2_2", "fantasy", "animate", "lynx"] or test_class_i2v(base_model_type) or test_wan_5B(base_model_type) or vace_class),  #"ti2v_2_2",
        "multiple_submodels" : multiple_submodels,
        "guidance_max_phases" : 3,
        "custom_sigmas": True,
        "custom_sigmas_continuous": True,
        "custom_sigmas_switch_scale": 1000.0,
        "custom_sigmas_schedulers": ["", "unipc", "euler", "dpm++"],
        "perturbation" : True,
        "flow_shift": True,
        "cfg_zero" : True,
        "cfg_star" : True,
        "adaptive_projected_guidance" : True,  
        "tea_cache" : not shotplan and not (base_model_type in ["i2v_2_2"] or test_wan_5B(base_model_type) or multiple_submodels),
        "mag_cache" : not shotplan,
        "keep_frames_video_guide_not_supported": base_model_type in ["infinitetalk"],
        "sample_solvers":[
                            ("unipc", "unipc"),
                            ("euler", "euler"),
                            ("dpm++", "dpm++"),
                            ("flowmatch causvid", "causvid"),
                            ("lcm + ltx", "lcm"), ]
        })
        extra_model_def["sub_parallel_windows"] = scail2

        extra_model_def["self_refiner"] = True

        if i2v:
            extra_model_def["motion_amplitude"] = True
 
            if base_model_type in ["i2v_2_2"]: 
                extra_model_def["i2v_v2v"] = True
                extra_model_def["extract_guide_from_window_start"] = True
                extra_model_def["guide_custom_choices"] = {
                    "choices":[("Use Text & Image Prompt Only", ""),
                            ("Video to Video guided by Text Prompt & Image", "GUV"),
                            ("Video to Video guided by Text/Image Prompt and Restricted to the Area of the Video Mask", "GVA")],
                    "default": "",
                    "show_label" : False,
                    "letters_filter": "GUVA",
                    "label": "Video to Video"
                }

                extra_model_def["mask_preprocessing"] = {
                    "selection":[ "", "A"],
                    "visible": False
                }
            if svi2pro:
                extra_model_def["image_ref_choices"] = {
                        "choices": [("No Anchor Image", ""),
                        ("Anchor Images For Each Window", "KI"),
                        ],
                        "letters_filter":  "KI",
                        "show_label" : False,
                }
                extra_model_def["all_image_refs_are_background_ref"] = True
                extra_model_def["no_background_removal"] = True
                extra_model_def["parent_model_type"] = "i2v_2_2"


        if base_model_type in ["i2v", "flf2v_720p"] or test_i2v_2_2(base_model_type):
            extra_model_def["black_frame"] = True
            

        if t2v: 
            if not alpha: 
                extra_model_def["guide_custom_choices"] = {
                    "choices":[("Use Text Prompt Only", ""),
                            ("Video to Video guided by Text Prompt", "GUV"),
                            ("Video to Video guided by Text Prompt and Restricted to the Area of the Video Mask", "GVA")],
                    "default": "",
                    "show_label" : False,
                    "letters_filter": "GUVA",
                    "label": "Video to Video"
                }

                extra_model_def["mask_preprocessing"] = {
                    "selection":[ "", "A"],
                    "visible": False
                }
            extra_model_def["v2i_switch_supported"] = True

        if bernini:
            extra_model_def.update({
                "guide_preprocessing": {
                    "selection": ["", "V"],
                    "labels": {"V": "Use Control Video"},
                    "default": "V",
                    "label": "Control Video Process",
                },
                "image_ref_choices": {
                    "choices": [("None", ""), ("Reference Images", "I")],
                    "letters_filter": "I",
                },
                "mask_preprocessing": {
                    "selection": ["", "A"],
                    "labels": {"": "Mask Not Used", "A": "Use Video Mask"},
                    "label": "Control Video Mask",
                },
                "return_image_refs_tensor": True,
                "no_background_removal": True,
                "fit_into_canvas_image_refs": 0,
                "control_video_trim": True,
                "control_net_weight_name": "Video Guidance",
                "control_net_weight_size": 1,
                "alt_guidance": "Reference Guidance",
                "sliding_window": False,
                "v2i_switch_supported": True,
                "tea_cache": False,
                "mag_cache": False,
                "self_refiner": False,
                "NAG": False,
                "cfg_star": False,
                "cfg_zero": False,
            })

        if base_model_type == "vista4d":
            extra_model_def.update({
                "guide_custom_choices": {
                    "choices": [("Use Vista4D control video", "UV")],
                    "default": "UV",
                    "visible": False,
                    "show_label": False,
                    "letters_filter": "UV",
                    "label": "Vista4D Control",
                },
                "model_modes": {
                    "label": "Vista4D Camera Movement",
                    "default": "dolly_zoom",
                    "image_modes": [0],
                    "choices": camera_infos.CAMERA_PRESET_CHOICES,
                },
                "custom_guide": {"id": "custom_guide", "name": "Custom Camera Movement", "label": "Custom Camera Movement (.npz)", "type": "file", "default": None, "required": False, "file_types": [".npz"]},
                "mask_preprocessing": {"selection": [""], "visible": False},
                "custom_settings": [
                    {"id": "vista4d_scene_scale", "name": "Scene scale", "label": "Vista4D scene scale", "type": "float", "default": 1.0},
                    {"id": "vista4d_camera_strength", "name": "Camera strength", "label": "Vista4D camera strength (%)", "type": "float", "default": 100.0},
                    {"id": "vista4d_seg_keywords", "name": "Dynamic object keywords", "label": "Dynamic object keywords", "type": "text", "default": "_all_", "info": "Comma-separated prompts; use _all_ for all pixels."},
                ],
                "control_video_trim": True,
                "keep_frames_video_guide_not_supported": True,
                "no_guide2_refresh": True,
                "preprocess_all": False,
                "sliding_window": False,
                "v2i_switch_supported": False,

            })


        if test_camera_control(base_model_type):
            fun_camera = base_model_type == "fun_camera_1.3B"
            prefix = "FUN_CAMERA" if fun_camera else "UNI3C"
            extra_model_def.update({key: getattr(camera_infos, f"{prefix}_{key.upper()}") for key in ("infos", "prompt_infos", "deepy_infos", "deepy_prompt_infos")})
            extra_model_def.update({
                "specialities": [camera_infos.CAMERA_CONTROL_SPECIALITY],
                "fun_camera": fun_camera,
                "uni3c": not fun_camera,
                "model_modes": {"label": "Camera Movement", "default": "arc_left_45", "image_modes": [0], "choices": camera_infos.CAMERA_PRESET_CHOICES},
                "custom_guide": camera_infos.CAMERA_PATH_GUIDE,
                "custom_settings": [camera_infos.CAMERA_STRENGTH_SETTING],
                "sliding_window": False,
                "motion_amplitude": False,
                "v2i_switch_supported": False,
            })

        if base_model_type in ["wanmove"]:
            extra_model_def["custom_guide"] = {"id": "custom_guide", "name": "Trajectory", "label": "Trajectory File (.npy)", "type": "file", "default": None, "required": True, "file_types": [".npy"]}
            extra_model_def["i2v_trajectory"] = True

        if base_model_type in ["steadydancer"]:
            extra_model_def["guide_custom_choices"] = {
            "choices":[
                ("Use Control Video Poses to Animate Person in Start Image", "V"),
                ("Use Control Video Poses filterd with Mask Video to Animate Person in Start Image", "VA"),
            ],
            "default": "PVB",
            "letters_filter": "PVBA",
            "label": "Type of Process",
            "scale": 3,
            "show_label" : False,
            }
            extra_model_def["custom_preprocessor"] = "Extracting Pose Information"
            extra_model_def["alt_guidance"] = "Condition Guidance"
            extra_model_def["no_guide2_refresh"] = True
            extra_model_def["no_mask_refresh"] = True
            extra_model_def["control_video_trim"] = True

        if base_model_type in ["scail"] or scail2:
            extra_model_def["guide_custom_choices"] = {
                "choices": [
                    ("Animate One Person", "V1"),
                    ("Animate Two Persons", "V2"),
                    ("Animate Three Persons", "V3"),
                    ("Animate Four Persons", "V4"),
                    ("Animate Five Persons", "V5"),
                    ("Replace One Person", "V01"),
                    ("Replace Two Persons", "V02"),
                    ("Replace Three Persons", "V03"),
                    ("Replace Four Persons", "V04"),
                    ("Replace Five Persons", "V05"),
                ] if scail2 else [
                    ("Animate One Person", "V#1#"),
                    ("Animate Two Persons", "V#2#"),
                    ("Animate Three Persons", "V#3#"),
                    ("Animate Four Persons", "V#4#"),
                    ("Animate Five Persons", "V#5#"),
                ],
                "default": "V1" if scail2 else "V#1#",
                "letters_filter": "V012345" if scail2 else "V#12345",
                "label": "Type of Process",
                "scale": 3,
                "show_label": True,
            }
            if scail2:
                extra_model_def["fake_start_image"] = True
                extra_model_def["fit_into_canvas_image_refs"] = 0            
            extra_model_def["preprocess_all"] = preprocess_all_scail2 if scail2 else True
            extra_model_def["custom_preprocessor"] = "Preparing Scail2 Inputs" if scail2 else "Extracting 3D Pose (NLFPose)"
            extra_model_def["forced_guide_mask_inputs"] = True
            extra_model_def["keep_frames_video_guide_not_supported"] = True
            extra_model_def["mask_preprocessing"] = {
                "selection": ["", "A", "NA"],
                "visible": True,
                "label": "Persons Locations"
            }
            extra_model_def["control_video_trim"] = True
            extra_model_def["extract_guide_from_window_start"] = True

            extra_model_def["return_image_refs_tensor"] = True
            if scail2:
                extra_model_def["infos"] = model_def.get("infos", SCAIL2_INFOS)
                extra_model_def["custom_image_ref_postprocessor"] = custom_image_ref_postprocessor_scail2
                extra_model_def["custom_settings"] = list(model_def.get("custom_settings", [])) if isinstance(model_def.get("custom_settings", []), list) else []
                extra_model_def["custom_settings"].append({
                    "id": "scail2_animate_preprocessing",
                    "name": "Animate preprocessing",
                    "label": "Animate Preprocessing",
                    "type": "dropdown",
                    "default": SCAIL2_ANIMATE_PREPROCESSING_RAW,
                    "choices": [
                        ("Use Raw Control Video Content", SCAIL2_ANIMATE_PREPROCESSING_RAW),
                        ("Extract 3D Pose information", SCAIL2_ANIMATE_PREPROCESSING_POSE),
                    ],
                    "video_prompt_type": "V",
                    "video_prompt_type_not": "0",
                })
                extra_model_def["custom_settings"].append({
                    "id": "image_ref_keyword_content",
                    "name": "Image Ref Keyword content",
                    "label": "Image Ref Keyword content",
                    "type": "text",
                    "default": "human character",
                    "video_prompt_type": "I",
                })
                extra_model_def["custom_settings"].append({
                    "id": SCAIL2_INJECT_REF_FRAMES_SETTING,
                    "name": "Inject Ref Frames in Video",
                    "label": "Inject Ref Frames in Video",
                    "type": "dropdown",
                    "default": SCAIL2_INJECT_REF_FRAMES_NO,
                    "choices": [
                        (SCAIL2_INJECT_REF_FRAMES_NO, SCAIL2_INJECT_REF_FRAMES_NO),
                        (SCAIL2_INJECT_REF_FRAMES_YES, SCAIL2_INJECT_REF_FRAMES_YES),
                    ],
                    "video_prompt_type": "I",
                })
                extra_model_def["image_ref_choices"] = {
                    "choices": [
                        ("None", ""),
                        ("Reference Images", "I"),
                    ],
                    "default": "",
                    "label": "Reference Images",
                    "letters_filter": "KI",
                    "visible": True,
                }
                extra_model_def["no_background_removal"] = True
                extra_model_def["magic_mask_object_colors"] = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0)]
                extra_model_def["video_mask_label"] = "Video Mask, each Colored Mask corresponds to a person to Animate"
                extra_model_def["custom_preprocessor_raw_inputs"] = True
                extra_model_def["video_mask_replace_background_color"] = [255, 255, 255]
                extra_model_def["background_removal_color"] = [255, 255, 255]
                extra_model_def["ref_matte_background_color"] = [0, 0, 0]

        if base_model_type in ["infinitetalk"]: 
            extra_model_def["no_background_removal"] = True
            extra_model_def["all_image_refs_are_background_ref"] = True
            extra_model_def["deepy_tools"] = {
                "gen_video_with_speech": {
                    "image_start": "image_refs",
                }
            }
            extra_model_def["guide_custom_choices"] = {
            "choices":[
                ("Images to Video, each Reference Image will start a new shot with a new Sliding Window", "KI"),
                ("Sparse Video to Video, one Image will by extracted from Video for each new Sliding Window", "RUV"),
                ("Video to Video, amount of motion transferred depends on Denoising Strength", "GUV"),
            ],
            "default": "KI",
            "letters_filter": "RGUVKI",
            "label": "Video to Video",
            "scale": 3,
            "show_label" : False,
            }

            extra_model_def["custom_video_selection"] = {
            "choices":[
                ("Smooth Transitions", ""),
                ("Sharp Transitions", "0"),
            ],
            "trigger": "",
            "label": "Custom Process",
            "letters_filter": "0",
            "show_label" : False,
            "scale": 1,
            }


            # extra_model_def["at_least_one_image_ref_needed"] = True
        if base_model_type == "lucy_edit":
            extra_model_def["keep_frames_video_guide_not_supported"] = True
            extra_model_def["guide_preprocessing"] = {
                    "selection": ["UV"],
                    "labels" : { "UV": "Control Video"},
                    "visible": False,
                }

        if base_model_type == "kiwi_edit":
            extra_model_def["keep_frames_video_guide_not_supported"] = True
            kiwi_ref_embedder = extra_model_def.get("kiwi_ref_embedder", True)
            extra_model_def["guide_custom_choices"] = {
            "choices":[
                ("Control", "UVI" if kiwi_ref_embedder else "UV"),
            ],
            "default": "UVI" if kiwi_ref_embedder else "UV",
            "letters_filter": "UVI",
            "label": "Type of Process",
            "scale": 3,
            "visible": False,
            "show_label" : False,
            }
            if kiwi_ref_embedder:
                extra_model_def["one_image_ref_needed"] = True

            # Upstream Kiwi uses full reference images; avoid background-matting path by default.
            extra_model_def["no_background_removal"] = True
            extra_model_def["fit_into_canvas_image_refs"] = 0
            extra_model_def["guidance_max_phases"] = 0            

        if animate2:
            extra_model_def.update({
                "guide_custom_choices": {
                    "choices": [("Animate Character using the Driving Video", "UV")],
                    "default": "UV",
                    "letters_filter": "UV",
                    "label": "Type of Process",
                    "visible": False,
                    "show_label": False,
                },
                "guide_preprocessing": {
                    "selection": ["UV"],
                    "labels": {"UV": "Driving Video"},
                    "visible": False,
                },
                "image_ref_choices": {
                    "choices": [("Use Start / Continue Frame", ""), ("Use Character Reference Image", "I")],
                    "default": "",
                    "letters_filter": "I",
                    "label": "Character Reference",
                    "show_label": True,
                },
                "alt_prompt": {
                    "label": "Driving Video Prompt",
                    "placeholder": "Describe the action and background in the driving video.",
                    "lines": 2,
                },
                "fake_start_image": True,
                "return_image_refs_tensor": True,
                "no_background_removal": True,
                "fit_into_canvas_image_refs": 1,
                "control_video_trim": True,
                "keep_frames_video_guide_not_supported": True,
                "extract_guide_from_window_start": True,
                "color_correction": False,
                "guidance_max_phases": 1,
                "perturbation": False,
                "tea_cache": False,
                "mag_cache": False,
                "self_refiner": False,
                "NAG": False,
                "cfg_star": False,
                "cfg_zero": False,
                "adaptive_projected_guidance": False,
                "sub_parallel_windows": False,
            })

        if base_model_type in ["animate"]:
            extra_model_def["guide_custom_choices"] = {
            "choices":[
                ("Animate Person in Reference Image using Motion of Whole Control Video", "PVBKI"),
                ("Animate Person in Reference Image using Motion of Targeted Person in Control Video", "PVBXAKI"),
                ("Replace Person in Control Video by Person in Ref Image", "PVBAIH#"),
                ("Replace Person in Control Video by Person in Ref Image. See Through Mask", "PVBAI#"),
            ],
            "default": "PVBKI",
            "letters_filter": "PVBXAKIH#",
            "label": "Type of Process",
            "scale": 3,
            "show_label" : False,
            }

            extra_model_def["custom_video_selection"] = {
            "choices":[
                ("None", ""),
                ("Apply Relighting", "1"),
            ],
            "trigger": "#",
            "label": "Custom Process",
            "type": "checkbox",
            "letters_filter": "1",
            "show_label" : False,
            "scale": 1,
            }

            extra_model_def["mask_preprocessing"] = {
                "selection":[ "", "A", "XA"],
                "visible": False
            }

            extra_model_def["video_guide_outpainting"] = [0,1]
            extra_model_def["keep_frames_video_guide_not_supported"] = True
            extra_model_def["extract_guide_from_window_start"] = True
            extra_model_def["forced_guide_mask_inputs"] = True
            extra_model_def["no_background_removal"] = True
            extra_model_def["background_removal_label"]= "Remove Backgrounds behind People (Animate Mode Only)"
            extra_model_def["background_ref_outpainted"] = False
            extra_model_def["return_image_refs_tensor"] = True
            extra_model_def["guide_inpaint_color"] = 0



        if vace_class:
            extra_model_def["control_net_weight_name"] = "Vace"
            extra_model_def["control_net_weight_size"] = 2
            extra_model_def["guide_preprocessing"] = {
                    "selection": ["", "UV", "PV", "OV", "DV", "SV", "LV", "CV", "MV", "V", "PDV", "PSV", "PLV" , "DSV", "DLV", "SLV"],
                    "labels" : { "V": "Use Vace raw format"}
                }
            extra_model_def["mask_preprocessing"] = {
                    "selection": ["", "A", "NA", "XA", "XNA", "YA", "YNA", "WA", "WNA", "ZA", "ZNA"],
                }

            extra_model_def["image_ref_choices"] = {
                    "choices": [("None", ""),
                    ("People / Objects", "I"),
                    ("Landscape followed by People / Objects (if any)", "KI"),
                    ("Positioned Frames followed by People / Objects (if any)", "FI"),
                    ],
                    "letters_filter":  "KFI",
            }

            extra_model_def["background_removal_label"]= "Remove Backgrounds behind People / Objects, keep it for Landscape or Positioned Frames"
            extra_model_def["video_guide_outpainting"] = [0,1]
            extra_model_def["pad_guide_video"] = True
            extra_model_def["guide_inpaint_color"] = 127.5
            extra_model_def["forced_guide_mask_inputs"] = True
            extra_model_def["return_image_refs_tensor"] = True
            extra_model_def["v2i_switch_supported"] = True
            if lynx:
                extra_model_def["set_video_prompt_type"]="Q"
                extra_model_def["control_net_weight_alt_name"] = "Lynx"
                extra_model_def["image_ref_choices"]["choices"] = [("None", ""),
                    ("People / Objects (if any) then a Face", "I"),
                    ("Landscape followed by People / Objects (if any) then a Face", "KI"),
                    ("Positioned Frames followed by People / Objects (if any) then a Face", "FI")]
                extra_model_def["background_removal_label"]= "Remove Backgrounds behind People / Objects, keep it for Landscape, Lynx Face or Positioned Frames"
                extra_model_def["no_processing_on_last_images_refs"] = 1
            if base_model_type in ["vace_ditto_14B"]:
                del extra_model_def["guide_preprocessing"], extra_model_def["image_ref_choices"], extra_model_def["video_guide_outpainting"]
                extra_model_def["mask_preprocessing"] = { "selection": ["", "A"], }
                extra_model_def["model_modes"] = {
                            "choices": [
                                ("Global", 0),
                                ("Global Style", 1),
                                ("Sim 2 Real", 2)],
                            "default": 0,
                            "label" : "Ditto Process"
                }

        if base_model_type in ["chrono_edit"]:
            extra_model_def["model_modes"] = {
                        "choices": [
                            ("Fast Image Transformation", 0),
                            ("Long Image Transformation", 1),
                            ("Temporal Reasoning Video", 2),],
                        "default": 0,
                        "label" : "Chrono Edit Process"
            }
            extra_model_def["custom_video_length"] = True


        if (not vace_class) and standin: 
            extra_model_def["v2i_switch_supported"] = True
            extra_model_def["image_ref_choices"] = {
                "choices": [
                    ("No Reference Image", ""),
                    ("Reference Image is a Person Face", "I"),
                    ],
                "visible": False,
                "letters_filter":"I",
            }
            extra_model_def["one_image_ref_needed"] = True

        if (not vace_class) and lynx: 
            extra_model_def["fit_into_canvas_image_refs"] = 0
            extra_model_def["guide_custom_choices"] = {
                "choices":[("Use Reference Image which is a Person Face", ""),
                           ("Video to Video guided by Text Prompt & Reference Image", "GUV"),
                           ("Video to Video on the Area of the Video Mask", "GVA")],
                "default": "",
                "letters_filter": "GUVA",
                "label": "Video to Video",
                "show_label" : False,
            }

            extra_model_def["mask_preprocessing"] = {
                "selection":[ "", "A"],
                "visible": False
            }

            extra_model_def["image_ref_choices"] = {
                "choices": [
                    ("No Reference Image", ""),
                    ("Reference Image is a Person Face", "I"),
                    ],
                "visible": False,
                "letters_filter":"I",
            }
            extra_model_def["one_image_ref_needed"] = True
            extra_model_def["set_video_prompt_type"]= "Q"
            extra_model_def["no_background_removal"] = True
            extra_model_def["v2i_switch_supported"] = True
            extra_model_def["control_net_weight_alt_name"] = "Lynx"


        if base_model_type in ["phantom_1.3B", "phantom_14B"]: 
            extra_model_def["image_ref_choices"] = {
                "choices": [("Reference Image", "I")],
                "letters_filter":"I",
                "visible": False,
            }

        if base_model_type in ["recam_1.3B"]: 
            extra_model_def["keep_frames_video_guide_not_supported"] = True
            extra_model_def["model_modes"] = {
                        "choices": [
                            ("Pan Right", 1),
                            ("Pan Left", 2),
                            ("Tilt Up", 3),
                            ("Tilt Down", 4),
                            ("Zoom In", 5),
                            ("Zoom Out", 6),
                            ("Translate Up (with rotation)", 7),
                            ("Translate Down (with rotation)", 8),
                            ("Arc Left (with rotation)", 9),
                            ("Arc Right (with rotation)", 10),
                        ],
                        "default": 1,
                        "label" : "Camera Movement Type"
            }
            extra_model_def["guide_preprocessing"] = {
                    "selection": ["UV"],
                    "labels" : { "UV": "Control Video"},
                    "visible" : False,
                }
            extra_model_def["video_length_locked"] = 81
        if base_model_type in ["chrono_edit"]:
            from .chono_edit_prompt import image_prompt_enhancer_instructions        
            extra_model_def["image_prompt_enhancer_instructions"] = image_prompt_enhancer_instructions
            extra_model_def["video_prompt_enhancer_instructions"] = image_prompt_enhancer_instructions
            extra_model_def["image_outputs"] = True
            extra_model_def["prompt_enhancer_choices_allowed"] = ["TI"]

        if bernini:
            image_prompt_types_allowed = "T"
        elif animate2:
            image_prompt_types_allowed = "TSVL"
        elif vace_class or base_model_type in ["animate", "t2v", "t2v_2_2", "shotplan_t2v", "shotplan_t2v_2_2", "lynx"] :
            image_prompt_types_allowed = "TVL"
        elif base_model_type in ["infinitetalk"]:
            image_prompt_types_allowed = "TSVL"
        elif base_model_type in ["ti2v_2_2"]:
            image_prompt_types_allowed = "TSVL"
        elif base_model_type in ["lucy_edit", "kiwi_edit"]:
            image_prompt_types_allowed = "TVL"
        elif scail2:
            image_prompt_types_allowed = "TSV"
        elif test_camera_control(base_model_type):
            image_prompt_types_allowed = "TS" if base_model_type == "fun_camera_1.3B" else "S"
        elif multitalk or base_model_type in ["fantasy", "steadydancer", "scail"]:
            image_prompt_types_allowed = "SVL"
        elif svi2pro:
            image_prompt_types_allowed = "SEVL"
        elif i2v:
            image_prompt_types_allowed = "SEVL"
        else:
            image_prompt_types_allowed = ""
        extra_model_def["image_prompt_types_allowed"] = image_prompt_types_allowed
        if base_model_type in ["mocha"]:
            extra_model_def["guide_custom_choices"] = {
            "choices":[
                ("Transfer Person In Reference Images (Second Image must be a Close Up) in Control Video", "VAI"),
            ],
            "default": "VAI",
            "letters_filter": "VAI",
            "label": "Type of Process",
            "scale": 3,
            "show_label" : False,
            "visible": True,
            }
            extra_model_def["background_removal_color"] = [128, 128, 128]  
        if base_model_type in ["fantasy"] or multitalk:
            extra_model_def["audio_guidance"] = True
        extra_model_def["NAG"] = (vace_class or t2v or i2v) and not (bernini or animate2)

        if test_oneframe_overlap(base_model_type):
            extra_model_def["sliding_window_defaults"] = { "overlap_min" : 1, "overlap_max" : 1, "overlap_step": 0, "overlap_default": 1}
        elif svi2pro:
            extra_model_def["sliding_window_defaults"] = { "overlap_min" : 4, "overlap_max" : 4, "overlap_step": 0, "overlap_default": 4, "overlap_offset": 0}

        # if base_model_type in ["phantom_1.3B", "phantom_14B"]: 
        #     extra_model_def["one_image_ref_needed"] = True


        guidance = wan_help(base_model_type, {**extra_model_def, **model_def})
        if base_model_type in ("vace_ditto_14B", "bernini_1.3B", "vace_standin_14B", "vace_lynx_14B", "vace_multitalk_14B"):
            extra_model_def["infos"] = guidance["infos"]
        result = merge_help(extra_model_def, guidance)
        result["infos"] += "\n\n**Custom Sigmas:** leave empty for the normal schedule. Separate values with commas and phase slots with `;`, using two semicolons for three phases. Blank phases keep normal schedule points within their switches. Each explicit later phase sets its switch from its first sigma. The complete schedule must descend across all phases. Supported schedulers: Euler, UniPC and DPM++. See the **i** beside the field for examples."
        result["deepy_infos"] += " `custom_sigmas` defaults to empty. Commas separate positive descending values below 1; use exactly `guidance_phases - 1` semicolons. Blank slots retain default schedule points in their phase range; trailing zero is ignored. This is one continuous schedule: explicit phase 2/3 sets `switch_threshold`/`switch_threshold2` to its first sigma times 1000, and every phase must stay within its boundaries. `num_inference_steps` builds the automatic schedule; explicit slots replace those points and change the actual total. Compatible schedulers: `euler`, `unipc`, `dpm++` and default."
        return result
        

    @staticmethod
    def get_rgb_factors(base_model_type ):
        from shared.RGB_factors import get_rgb_factors
        if test_wan_5B(base_model_type): base_model_type = "ti2v_2_2"
        if base_model_type == "animate2": base_model_type = "i2v"
        latent_rgb_factors, latent_rgb_factors_bias = get_rgb_factors("wan", base_model_type)
        return latent_rgb_factors, latent_rgb_factors_bias
    
    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        download_def = []
        if base_model_type == "kiwi_edit":
            kiwi_mllm_repo_id = model_def.get("kiwi_mllm_repo_id", "DeepBeepMeep/Wan2.2")
            kiwi_mllm_folder = model_def.get("kiwi_mllm_folder", "kiwi_mllm_encoder_instruct_reference")
            kiwi_mllm_files = ["config.json", "diffusion_pytorch_model.safetensors", "added_tokens.json", "chat_template.jinja", "merges.txt", "preprocessor_config.json", "qwen_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json"]
            download_def.append({"repoId": kiwi_mllm_repo_id, "sourceFolderList": [kiwi_mllm_folder], "fileList": [kiwi_mllm_files]})
            embedder_files = [f for f in (model_def.get("kiwi_source_embedder_file"), model_def.get("kiwi_ref_embedder_file")) if f]
            if embedder_files: download_def.append({"repoId": kiwi_mllm_repo_id, "sourceFolderList": [""], "fileList": [embedder_files]})
        else:
            download_def  += [{
                "repoId" : "DeepBeepMeep/Wan2.1", 
                "sourceFolderList" :  ["umt5-xxl" ],
                "fileList" : [ ["special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json"] ]   
            }]

        if test_wan_5B(base_model_type):
            download_def += [    {
                "repoId" : "DeepBeepMeep/Wan2.2", 
                "sourceFolderList" :  [""],
                "fileList" : [ [ "Wan2.2_VAE.safetensors"]  ]
            }]
        else:
            wan_files = ["Wan2.1_VAE.safetensors", "Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors"]
            if base_model_type in ["fantasy"]:
                wan_files.append("fantasy_proj_model.safetensors")
            download_def += [{
                "repoId" : "DeepBeepMeep/Wan2.1", 
                "sourceFolderList" :  ["xlm-roberta-large", ""  ],
                "fileList" : [ [ "models_clip_open-clip-xlm-roberta-large-vit-huge-14-bf16.safetensors", "sentencepiece.bpe.model", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"], wan_files ]   
            }]

        if base_model_type == "scail" or test_scail2(base_model_type):
            download_def += [
                {
                    "repoId": "DeepBeepMeep/Wan2.1",
                    "sourceFolderList": ["pose"],
                    "fileList": [["nlf_l_multi_0.3.2.eager.safetensors", "nlf_l_multi_0.3.2.eager.meta.json"]],
                }
            ]
            if test_scail2(base_model_type):
                from shared import magic_mask

                download_def.append(magic_mask.query_download_def())
        elif base_model_type == "vista4d":
            from preprocessing.depth_anything_v3.assets import query_download_def as query_depth_files
            from preprocessing.sam3.assets import query_download_def as query_sam3_files

            download_def += [query_depth_files(), query_sam3_files()]
        elif base_model_type == "uni3c":
            from preprocessing.depth_anything_v3.assets import query_download_def as query_depth_files

            download_def.append(query_depth_files(metric=True))

        return download_def

    @staticmethod
    def custom_preprocess(base_model_type, video_guide, video_mask, pre_video_guide=None,  max_workers = 1, expand_scale = 0, video_prompt_type = None, **kwargs):
        from shared.utils.utils import convert_tensor_to_image

        if test_scail2(base_model_type):
            return custom_preprocess_scail2(video_guide, video_mask, pre_video_guide=pre_video_guide, max_workers=max_workers, expand_scale=expand_scale, video_prompt_type=video_prompt_type, **kwargs)

        ref_image = convert_tensor_to_image(pre_video_guide[:, 0])
        frames = video_guide
        pose_mask = kwargs.get("pose_mask", None)
        mask_frames = pose_mask if pose_mask is not None else video_mask

        if base_model_type == "scail":
            def extract_max_people(s):
                return int(m.group(1)) if (m := re.search(r'#(\d+)#', s or "")) else 1

            # ref_image = ref_image.resize( (ref_image.width // 2, ref_image.height // 2), resample=Image.LANCZOS )
            from .scail import ScailPoseProcessor
            scail_max_people = extract_max_people(video_prompt_type)
            scail_multi_person = scail_max_people > 1
            processor = ScailPoseProcessor(multi_person=scail_multi_person, max_people=scail_max_people)
            video_guide_processed = processor.extract_and_render(
                frames,
                ref_image=ref_image,
                mask_frames=mask_frames,
                align_pose=True
            )
            if video_guide_processed.numel() == 0:
                gr.Info("Unable to detect a Person")
                return None, None, None, None
            return video_guide_processed, None, video_mask, None
        else:
            # Steadydancer 
            from .steadydancer.pose_align import PoseAligner
            aligner = PoseAligner()
            outputs = aligner.align(frames, ref_image, ref_video_mask=None, align_frame=0, max_frames=None, augment=True, include_composite=False, cpu_resize_workers=max_workers, expand_scale=expand_scale)

            video_guide_processed, video_guide_processed2 = outputs["pose_only"], outputs["pose_aug"]
            if video_guide_processed.numel() == 0:
                return None, None, None, None

            return video_guide_processed, video_guide_processed2, None, None 


    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, quantizeTransformer = False, text_encoder_quantization = None, dtype = torch.bfloat16, VAE_dtype = torch.float32, mixed_precision_transformer = False, save_quantized= False, submodel_no_list = None, text_encoder_filename = None, VAE_upsampling = None, **kwargs):
        from .configs import WAN_CONFIGS

        if test_class_i2v(base_model_type):
            cfg = WAN_CONFIGS['i2v-14B']
        else:
            cfg = WAN_CONFIGS['t2v-14B']
        from . import WanAny2V
        wan_model = WanAny2V(
            config=cfg,
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            submodel_no_list = submodel_no_list,
            model_type = model_type,        
            model_def = model_def,
            base_model_type=base_model_type,
            text_encoder_filename= text_encoder_filename,
            quantizeTransformer = quantizeTransformer,
            dtype = dtype,
            VAE_dtype = VAE_dtype, 
            mixed_precision_transformer = mixed_precision_transformer,
            save_quantized = save_quantized,
            VAE_upsampling = VAE_upsampling,            
        )

        pipe = {"transformer": wan_model.model, "vae": wan_model.vae.model}
        if base_model_type == "kiwi_edit":
            if getattr(wan_model, "kiwi_mllm", None) is None:
                raise RuntimeError("Kiwi MLLM encoder is required but not initialized.")
            wan_model.kiwi_mllm.prepare_for_mmgp()
            if wan_model.kiwi_mllm.encoder is None:
                raise RuntimeError("Kiwi MLLM encoder failed to load.")
            pipe["text_encoder"] = wan_model.kiwi_mllm.encoder
        else:
            pipe["text_encoder"] = wan_model.text_encoder.model
        if wan_model.vae2 is not None:
            pipe["vae2"] = wan_model.vae2.model             
        if hasattr(wan_model,"model2") and wan_model.model2 is not None:
            pipe["transformer2"] = wan_model.model2
        if hasattr(wan_model, "clip"):
            pipe["text_encoder_2"] = wan_model.clip.model
        return wan_model, pipe

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if ui_defaults.get("sample_solver", "") == "": 
            ui_defaults["sample_solver"] = "unipc"
        
        if settings_version < 2.24:
            if (model_def.get("multiple_submodels", False) or ui_defaults.get("switch_threshold", 0) > 0) and ui_defaults.get("guidance_phases",0)<2:
                ui_defaults["guidance_phases"] = 2

        if settings_version == 2.24 and ui_defaults.get("guidance_phases",0) ==2:
            mult = model_def.get("loras_multipliers","")
            if len(mult)> 1 and len(mult[0].split(";"))==3: ui_defaults["guidance_phases"] = 3

        if settings_version < 2.27:
            if base_model_type in "infinitetalk":
                guidance_scale = ui_defaults.get("guidance_scale", None)
                if guidance_scale == 1:
                    ui_defaults["audio_guidance_scale"]= 1
                video_prompt_type = ui_defaults.get("video_prompt_type", "")
                if "I" in video_prompt_type:
                    video_prompt_type = video_prompt_type.replace("KI", "0KI")
                    ui_defaults["video_prompt_type"] = video_prompt_type 

        if settings_version < 2.28:
            if base_model_type in "infinitetalk":
                video_prompt_type = ui_defaults.get("video_prompt_type", "")
                if "U" in video_prompt_type:
                    video_prompt_type = video_prompt_type.replace("U", "RU")
                    ui_defaults["video_prompt_type"] = video_prompt_type 

        if settings_version < 2.31:
            if base_model_type in ["recam_1.3B"]:
                video_prompt_type = ui_defaults.get("video_prompt_type", "")
                if not "V" in video_prompt_type:
                    video_prompt_type += "UV"
                    ui_defaults["video_prompt_type"] = video_prompt_type 
                    ui_defaults["image_prompt_type"] = ""

            if test_oneframe_overlap(base_model_type):
                ui_defaults["sliding_window_overlap"] = 1

        if settings_version < 2.32:
            image_prompt_type = ui_defaults.get("image_prompt_type", "")
            if test_class_i2v(base_model_type) and len(image_prompt_type) == 0 and "S" in model_def.get("image_prompt_types_allowed",""):
                ui_defaults["image_prompt_type"] = "S" 


        if settings_version < 2.37:
            if base_model_type in ["animate"]:
                video_prompt_type = ui_defaults.get("video_prompt_type", "")
                if "1" in video_prompt_type:
                    video_prompt_type = video_prompt_type.replace("1", "#1")
                    ui_defaults["video_prompt_type"] = video_prompt_type 

        if settings_version < 2.38:
            if base_model_type in ["infinitetalk"]:
                video_prompt_type = ui_defaults.get("video_prompt_type", "")
                if "Q" in video_prompt_type:
                    video_prompt_type = video_prompt_type.replace("Q", "0")
                    ui_defaults["video_prompt_type"] = video_prompt_type 

        if settings_version < 2.39:
            if base_model_type in ["fantasy"]:
                audio_prompt_type = ui_defaults.get("audio_prompt_type", "")
                if not "A" in audio_prompt_type:
                    audio_prompt_type +=  "A"
                    ui_defaults["audio_prompt_type"] = audio_prompt_type 

        if settings_version < 2.40:
            if base_model_type in ["animate"]:
                remove_background_images_ref = ui_defaults.get("remove_background_images_ref", None)
                if remove_background_images_ref !=0:
                    ui_defaults["remove_background_images_ref"] = 0

        if settings_version < 2.42 and test_svi2pro(base_model_type):
            ui_defaults.update({
                "sliding_window_size": 81, 
                "sliding_window_overlap" : 4,
            })

        if model_def.get("self_refiner",False) and settings_version < 2.47:
            ui_defaults["self_refiner_setting"] = 0
            ui_defaults["self_refiner_plan"] = ""
        if model_def.get("self_refiner",False) and settings_version < 2.48:
            ui_defaults["self_refiner_f_uncertainty"] = 0.1
            ui_defaults["self_refiner_certain_percentage"] = 0.999
        if settings_version < 2.52:
            plan = ui_defaults.get("self_refiner_plan")
            if isinstance(plan, list):
                from shared.utils.self_refiner import convert_refiner_list_to_string
                ui_defaults["self_refiner_plan"] = convert_refiner_list_to_string(plan)

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "sample_solver": "unipc",
        })
        if test_class_i2v(base_model_type) and "S" in model_def["image_prompt_types_allowed"]:
            ui_defaults["image_prompt_type"] = "S" 

        if base_model_type in ["fantasy"]:
            ui_defaults.update({
                "audio_guidance_scale": 5.0,
                "sliding_window_overlap" : 1,
                "audio_prompt_type": "A",
            })

        elif base_model_type in ["multitalk"]:
            ui_defaults.update({
                "guidance_scale": 5.0,
                "flow_shift": 7, # 11 for 720p
                "sliding_window_discard_last_frames" : 4,
                "sample_solver" : "euler",
                "audio_prompt_type": "A",
                "adaptive_switch" : 1,
            })

        elif base_model_type in ["infinitetalk"]:
            ui_defaults.update({
                "guidance_scale": 5.0,
                "flow_shift": 7, # 11 for 720p
                "sliding_window_overlap" : 9,
                "sliding_window_size": 81, 
                "sample_solver" : "euler",
                "video_prompt_type": "0KI",
                "remove_background_images_ref" : 0,
                "adaptive_switch" : 1,
            })

        elif base_model_type in ["standin"]:
            ui_defaults.update({
                "guidance_scale": 5.0,
                "flow_shift": 7, # 11 for 720p
                "sliding_window_overlap" : 9,
                "video_prompt_type": "I",
                "remove_background_images_ref" : 1 ,
            })

        elif (base_model_type in ["lynx_lite", "lynx", "alpha_lynx"]):
            ui_defaults.update({
                "guidance_scale": 5.0,
                "flow_shift": 7, # 11 for 720p
                "sliding_window_overlap" : 9,
                "video_prompt_type": "I",
                "denoising_strength": 0.8,
                "remove_background_images_ref" :  0,
            })

        elif base_model_type in ["phantom_1.3B", "phantom_14B"]:
            ui_defaults.update({
                "guidance_scale": 7.5,
                "flow_shift": 5,
                "remove_background_images_ref": 1,
                "video_prompt_type": "I",
                # "resolution": "1280x720" 
            })

        elif base_model_type in ["vace_14B", "vace_multitalk_14B"]:
            ui_defaults.update({
                "sliding_window_discard_last_frames": 0,
            })

        elif base_model_type in ["ti2v_2_2", "kiwi_edit"]:
            ui_defaults.update({
                "image_prompt_type": "T", 
            })
            if base_model_type == "kiwi_edit":
                ui_defaults.update({
                    "remove_background_images_ref": 0,
                    "guidance_scale": 1,
                })
                
                ui_defaults.update({
                    "video_prompt_type": "UVI" if model_def.get("kiwi_ref_embedder", True) else "UV", 
                })

        elif test_camera_control(base_model_type):
            fun_camera = base_model_type == "fun_camera_1.3B"
            ui_defaults.update({
                "video_length": 81,
                "num_inference_steps": 50,
                "guidance_scale": 6.0 if fun_camera else 5.0,
                "flow_shift": 3.0 if fun_camera else 5.0,
                "model_mode": "arc_left_45",
            })

        elif test_bernini(base_model_type):
            bernini_defaults = {
                "video_prompt_type": "VI",
                "resolution": "832x480",
                "video_length": 81,
                "num_inference_steps": 40,
                "flow_shift": 5,
                "guidance_scale": 4,
                "control_net_weight": 1.25,
                "alt_guidance_scale": 4.5,
                "sample_solver": "unipc",
                "remove_background_images_ref": 0,
                "prompt_enhancer": "",
            }
            if base_model_type == "bernini_1.3B":
                bernini_defaults.update({"guidance_phases": 1, "switch_threshold": 0})
            else:
                bernini_defaults.update({"guidance_phases": 2, "model_switch_phase": 1, "switch_threshold": 875, "guidance2_scale": 4, "guidance3_scale": 1})
            ui_defaults.update(bernini_defaults)

        if base_model_type in ["recam_1.3B", "lucy_edit"]: 
            ui_defaults.update({
                "video_prompt_type": "UV", 
            })
        elif base_model_type in ["animate"]: 
            ui_defaults.update({ 
                "video_prompt_type": "PVBKI", 
                "mask_expand": 20,
                "audio_prompt_type": "R",
                "remove_background_images_ref" : 0,
                "force_fps": "control",
            })
        elif base_model_type == "animate2":
            ui_defaults.update({
                "video_prompt_type": "UV",
                "image_prompt_type": "S",
                "video_length": 65,
                "sliding_window_size": 65,
                "num_inference_steps": 40,
                "flow_shift": 5,
                "guidance_scale": 3.0,
                "sample_solver": "dpm++",
                "remove_background_images_ref": 0,
                "force_fps": "control",
                "alt_prompt": model_def["animate2_ref_prompt"],
                "prompt_enhancer": "",
            })
        elif base_model_type in ["vace_ditto_14B"]:
            ui_defaults.update({ 
                "video_prompt_type": "V", 
            })
        elif base_model_type in ["mocha"]:
            ui_defaults.update({ 
                "video_prompt_type": "VAI", 
                "audio_prompt_type": "R",
	            "force_fps": "control",
            })
        elif base_model_type in ["steadydancer"]:
            ui_defaults.update({
                "video_prompt_type": "VA",
                "image_prompt_type": "S",
                "audio_prompt_type": "R",
                "force_fps": "control",
                "alt_guidance_scale" : 2.0,
            })
        elif base_model_type in ["scail"]:
            ui_defaults.update({
                "video_prompt_type": "V#1#",
                "image_prompt_type": "S",
                "audio_prompt_type": "R",
                "force_fps": "control",
                "sliding_window_overlap" : 1,
                "sliding_window_size": 81,
            })
        elif test_scail2(base_model_type):
            ui_defaults.update({
                "video_prompt_type": "V1",
                "image_prompt_type": "S",
                "audio_prompt_type": "R",
                "force_fps": "control",
                "sliding_window_overlap" : 5,
                "sliding_window_size": 81,
                "num_inference_steps": 40,
                "flow_shift": 3,
                "guidance_scale": 5.0,
                "remove_background_images_ref": 0,
            })

        if test_svi2pro(base_model_type):
            ui_defaults.update({
                "sliding_window_size": 81, 
                "sliding_window_overlap" : 4,
            })
            
        if test_wan_5B(base_model_type):
            ui_defaults.update({
                "sliding_window_size": 121, 
            })

        if base_model_type in ["i2v_2_2"]:
            ui_defaults.update({"masking_strength": 0.1, "denoising_strength": 0.9})
            
        if base_model_type in ["chrono_edit"]:
            ui_defaults.update({"image_mode": 1, "prompt_enhancer":"TI"})

        if test_oneframe_overlap(base_model_type):
            ui_defaults["sliding_window_overlap"] = 1
            ui_defaults["sliding_window_color_correction_strength"]= 0

        if test_multitalk(base_model_type):
            ui_defaults["audio_guidance_scale"] = 4

        if model_def.get("multiple_submodels", False):
            ui_defaults["guidance_phases"] = 2
    
    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if base_model_type == "animate2":
            image_prompt_type = inputs.get("image_prompt_type", "") or ""
            video_prompt_type = inputs.get("video_prompt_type", "") or ""
            if "I" not in video_prompt_type and not any(flag in image_prompt_type for flag in "SVL"):
                return "Wan Animate 2 needs a Character Reference Image, Start Image, Continue Video, or Continue Last Video."

        elif test_scail2(base_model_type):
            image_prompt_type = inputs.get("image_prompt_type", "") or ""
            video_prompt_type = inputs.get("video_prompt_type", "") or ""
            if test_scail2_replace(video_prompt_type):
                if any(flag in image_prompt_type for flag in "SVL"):
                    return "SCAIL-2 Replace mode uses the Control Video as the source; set Location to Text Prompt."
                if inputs.get("image_refs", None) is None:
                    return "SCAIL-2 Replace mode requires a Reference Image."
                if "A" not in video_prompt_type or "N" in video_prompt_type:
                    return "SCAIL-2 Replace mode requires Persons Locations set to Masked Area."
                if inputs.get("video_mask", None) is None:
                    return "SCAIL-2 Replace mode requires a Video Mask."
            elif not any(flag in image_prompt_type for flag in "SV"):
                return "SCAIL-2 Animate mode requires Location set to Start Image or Continue Video."

        elif base_model_type in ["infinitetalk"]:
            video_source = inputs["video_source"]
            image_refs = inputs["image_refs"]
            video_prompt_type = inputs["video_prompt_type"]
            image_prompt_type = inputs["image_prompt_type"]
            if ("V" in image_prompt_type or "L" in image_prompt_type) and image_refs is None:
                video_prompt_type = video_prompt_type.replace("I", "").replace("K","")
                inputs["video_prompt_type"] = video_prompt_type 


        elif base_model_type in ["vace_standin_14B", "vace_lynx_14B"]:
            image_refs = inputs["image_refs"]
            video_prompt_type = inputs["video_prompt_type"]
            if image_refs is not None and len(image_refs) == 1 and "K" in video_prompt_type:
                gr.Info("Warning, Ref Image that contains the Face to transfer is Missing: if 'Landscape and then People or Objects' is selected beside the Landscape Image Ref there should be another Image Ref that contains a Face.")
                    

        elif base_model_type in ["chrono_edit"]:
            model_mode = inputs["model_mode"]
            inputs["video_length"] = 5 if model_mode==0 else 29
            inputs["image_mode"] = 0 if model_mode==2 else 1
