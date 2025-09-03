#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def main():
    # 定义划分
    object_split = {
        '45oz_RAMEKIN_ASST_DEEP_COLORS': 'Train',
        'ACE_Coffee_Mug_Kristen_16_oz_cup': 'Train',
        'Aroma_Stainless_Steel_Milk_Frother_2_Cup': 'Train',
        'BIA_Cordon_Bleu_White_Porcelain_Utensil_Holder_900028': 'Train',
        'BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup': 'Test',
        'Black_Decker_Stainless_Steel_Toaster_4_Slice': 'Train',
        'Black_and_Decker_TR3500SD_2Slice_Toaster': 'Train',
        'Bradshaw_International_11642_7_Qt_MP_Plastic_Bowl': 'Train',
        'Calphalon_Kitchen_Essentials_12_Cast_Iron_Fry_Pan_Black': 'Train',
        'Central_Garden_Flower_Pot_Goo_425': 'Test',
        'Chef_Style_Round_Cake_Pan_9_inch_pan': 'Test',
        'Chefmate_8_Frypan': 'Train',
        'Cole_Hardware_Bowl_Scirocco_YellowBlue': 'Test',
        'Cole_Hardware_Butter_Dish_Square_Red': 'Train',
        'Cole_Hardware_Deep_Bowl_Good_Earth_1075': 'Test',
        'Cole_Hardware_Electric_Pot_Assortment_55': 'Train',
        'Cole_Hardware_Electric_Pot_Cabana_55': 'Train',
        'Cole_Hardware_Flower_Pot_1025': 'Test',
        'Cole_Hardware_Mug_Classic_Blue': 'Train',
        'Cole_Hardware_Orchid_Pot_85': 'Train',
        'Corningware_CW_by_Corningware_3qt_Oblong_Casserole_Dish_Blue': 'Test',
        'Dixie_10_ounce_Bowls_35_ct': 'Train',
        'Don_Franciscos_Gourmet_Coffee_Medium_Decaf_100_Colombian_12_oz_340_g': 'Train',
        'Down_To_Earth_Ceramic_Orchid_Pot_Asst_Blue': 'Train',
        'Down_To_Earth_Orchid_Pot_Ceramic_Lime': 'Train',
        'Down_To_Earth_Orchid_Pot_Ceramic_Red': 'Test',
        'Ecoforms_Cup_B4_SAN': 'Train',
        'Ecoforms_Garden_Pot_GP16ATurquois': 'Train',
        'Ecoforms_Plant_Bowl_Atlas_Low': 'Train',
        'Ecoforms_Plant_Bowl_Turquoise_7': 'Train',
        'Ecoforms_Plant_Container_12_Pot_Nova': 'Test',
        'Ecoforms_Plant_Plate_S11Turquoise': 'Train',
        'Ecoforms_Plant_Pot_GP9AAvocado': 'Train',
        'Ecoforms_Plant_Pot_GP9_SAND': 'Train',
        'Ecoforms_Planter_Bowl_Cole_Hardware': 'Train',
        'Ecoforms_Planter_Pot_GP12AAvocado': 'Train',
        'Ecoforms_Planter_Pot_QP6Ebony': 'Test',
        'Ecoforms_Plate_S20Avocado': 'Test',
        'Ecoforms_Pot_Nova_6_Turquoise': 'Train',
        'Footed_Bowl_Sand': 'Train',
        'Little_Debbie_Chocolate_Cupcakes_8_ct': 'Train',
        'Nescafe_Tasters_Choice_Instant_Coffee_Decaf_House_Blend_Light_7_oz': 'Test',
        'Nestl_Crunch_Girl_Scouts_Cookie_Flavors_Caramel_Coconut_78_oz_box': 'Train',
        'Nestle_Carnation_Cinnamon_Coffeecake_Kit_1913OZ': 'Train',
        'Nordic_Ware_Original_Bundt_Pan': 'Train',
        'Now_Designs_Bowl_Akita_Black': 'Train',
        'OXO_Cookie_Spatula': 'Train',
        'Pennington_Electric_Pot_Cabana_4': 'Train',
        'Room_Essentials_Bowl_Turquiose': 'Test',
        'Room_Essentials_Dish_Drainer_Collapsible_White': 'Train',
        'Room_Essentials_Mug_White_Yellow': 'Test',
        'Room_Essentials_Salad_Plate_Turquoise': 'Train',
        'Sapota_Threshold_4_Ceramic_Round_Planter_Red': 'Train',
        'Sea_to_Summit_Xl_Bowl': 'Train',
        'Threshold_Bead_Cereal_Bowl_White': 'Test',
        'Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring': 'Test',
        'Threshold_Dinner_Plate_Square_Rim_White_Porcelain': 'Train',
        'Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White': 'Train',
        'Threshold_Porcelain_Pitcher_White': 'Train',
        'Threshold_Porcelain_Serving_Bowl_Coupe_White': 'Train',
        'Threshold_Porcelain_Spoon_Rest_White': 'Train',
        'Threshold_Porcelain_Teapot_White': 'Train',
        'Threshold_Ramekin_White_Porcelain': 'Train',
        'Threshold_Salad_Plate_Square_Rim_Porcelain': 'Train',
        'Threshold_Tray_Rectangle_Porcelain': 'Test',
        'TriStar_Products_PPC_Power_Pressure_Cooker_XL_in_Black': 'Train',
        'Twinlab_100_Whey_Protein_Fuel_Cookies_and_Cream': 'Test',
        'Utana_5_Porcelain_Ramekin_Large': 'Test'
    }
    
    # 源目录和目标目录
    source_dir = Path("data/urdfs/mess_kitchen")
    train_dir = Path("data/urdfs/mess_kitchen/train")
    test_dir = Path("data/urdfs/mess_kitchen/test")
    
    # 创建目标目录
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计
    train_count = 0
    test_count = 0
    missing_count = 0
    
    print("开始整理URDF文件...")
    
    # 遍历所有URDF文件
    for urdf_file in source_dir.glob("*.urdf"):
        object_name = urdf_file.stem  # 去掉.urdf扩展名
        
        if object_name in object_split:
            split_type = object_split[object_name]
            
            if split_type == 'Train':
                target_path = train_dir / urdf_file.name
                shutil.move(str(urdf_file), str(target_path))
                print(f"✓ {object_name} -> train/")
                train_count += 1
            elif split_type == 'Test':
                target_path = test_dir / urdf_file.name
                shutil.move(str(urdf_file), str(target_path))
                print(f"✓ {object_name} -> test/")
                test_count += 1
        else:
            print(f"⚠ {object_name} 不在划分列表中")
            missing_count += 1
    
    print(f"\n整理完成！")
    print(f"Train: {train_count} 个文件")
    print(f"Test: {test_count} 个文件")
    print(f"Missing: {missing_count} 个文件")
    print(f"Train目录: {train_dir}")
    print(f"Test目录: {test_dir}")

if __name__ == "__main__":
    main()
