import json
import csv
import re
import time


# 抽取个人信息：年龄、性别、婚姻状态
def personal_information(data):
    if 'admissions_records' in data:
        try:
            age = int(data['admissions_records'][0]['AGE'])
        except:
            age = ''
        try:
            gender = data['admissions_records'][0]['GENDER']
        except:
            gender = ""
        try:
            marital_status = data['admissions_records'][0]['MARITAL_STATUS']
        except:
            marital_status = ""
    else:
        age = ""
        gender = ""
        marital_status = ""
    return age, gender, marital_status


# 抽取主诉
def Chief_Complaint(data):
    if 'admissions_records' in data:
        try:
            chief_complaint = data['admissions_records'][0]['CHIEF_COMPLAINT']
        except:
            chief_complaint = ""
    else:
        chief_complaint = ""
    # 有些主诉不规范，把现病史也放进来了，需要剔除，主诉一般在第一句话，不超过30个字符
    #     if len(chief_complaint) > 30:
    #         chief_complaint = chief_complaint[:30]
    return chief_complaint


# 抽取生命体征：体温、脉搏、呼吸、血压
def vital_signs(data):
    if 'admissions_records' in data:
        if 'PHYSICAL_EXAMINATION' in data['admissions_records'][0]:
            # 生命体征在physical examination的文本中，一般在段首，50个字符以内。
            text = data['admissions_records'][0]['PHYSICAL_EXAMINATION'][:50]
            # 抽取体温
            try:
                key = '[Tt体温].*?\d+(\.?)\d*'  # 正则表达式，搜寻含有T,t,体温字样的数字内容，以下同样。
                find = re.search(re.compile(key), text).group(0)
                key_ = '\d+(\.?)\d*'  # 去掉其他字符，只保留数字
                temperature = re.search(re.compile(key_), find).group(0)
            except:
                temperature = ''
            # 抽取脉搏
            try:
                key = '[Pp脉搏].*?\d+'
                find = re.search(re.compile(key), text).group(0)
                key_ = '\d+'
                pulse = re.search(re.compile(key_), find).group(0)
            except:
                pulse = ""
            # 抽取呼吸
            try:
                key = '[Rr呼吸].*?\d+'
                find = re.search(re.compile(key), text).group(0)
                key_ = '\d+'
                breath = re.search(re.compile(key_), find).group(0)
            except:
                breath = ""
            # 抽取血压
            try:
                key = '[bB][Pp].*?\d+(.+?)\d*'
                find = re.search(re.compile(key), text).group(0)
                key_ = '\d+'
                BP = re.compile(key_).findall(find)
                HBP = max(int(x) for x in BP)  # 收缩压
                LBP = min(int(x) for x in BP)  # 舒张压
            except:
                HBP = ""
                LBP = ""
        else:
            temperature, pulse, breath, HBP, LBP = '', '', '', '', ''

    else:
        temperature, pulse, breath, HBP, LBP = '', '', '', '', ''
    return temperature, pulse, breath, HBP, LBP


# 抽取化验：血常规和生化检查
def inspection_reports(data):
    if 'inspection_reports' in data:
        biochemistry = {}
        blood_routine = {}
        for inspection in data['inspection_reports']:
            # 只取第一次检查
            if len(biochemistry) == 0 and inspection['INSPECTION_NAME'] == '生化系列(35项)':
                try:
                    sub_inspection = inspection['sub_inspection']
                    for item in sub_inspection:
                        name = item['SUB_INSPECTION_EN']
                        result = float(item['SUB_INSPECTION_RESULT'])
                        biochemistry[name] = result
                except:
                    pass
        try:
            A_G, BUN_CREA, AST_ALT = biochemistry['A:G'], biochemistry['BUN:CREA'], biochemistry['AST:ALT']
            ALB, ALP, ALT, ANION = biochemistry['ALB'], biochemistry['ALP'], biochemistry['ALT'], biochemistry['ANION']
            AST, BUN, CA, CHOL = biochemistry['AST'], biochemistry['BUN'], biochemistry['CA'], biochemistry['CHOL']
            CK, Cl, CO2, CREA = biochemistry['CK'], biochemistry['Cl'], biochemistry['CO2'], biochemistry['CREA']
            DBIL, GGT, GLO, GLU = biochemistry['DBIL'], biochemistry['GGT'], biochemistry['GLO'], biochemistry['GLU']
            HDL_C, IBIL, K, LDH = biochemistry['HDL-C'], biochemistry['IBIL'], biochemistry['K'], biochemistry['LDH']
            LDL_C, Lp, MG, Na = biochemistry['LDL-C'], biochemistry['Lp(a)'], biochemistry['MG'], biochemistry['Na']
            OSM, Pi, SA, TBA = biochemistry['OSM'], biochemistry['Pi'], biochemistry['SA'], biochemistry['TBA']
            TBIL, TG, TP, URIC = biochemistry['TBIL'], biochemistry['TG'], biochemistry['TP'], biochemistry['URIC']
        except:
            A_G, BUN_CREA, AST_ALT, ALB, ALP, ALT, ANION = '', '', '', '', '', '', ''
            AST, BUN, CA, CHOL, CK, Cl, CO2, CREA = '', '', '', '', '', '', '', ''
            DBIL, GGT, GLO, GLU, HDL_C, IBIL, K, LDH = '', '', '', '', '', '', '', ''
            LDL_C, Lp, MG, Na, OSM, Pi, SA, TBA = '', '', '', '', '', '', '', ''
            TBIL, TG, TP, URIC = '', '', '', ''

        for inspection in data['inspection_reports']:
            if len(blood_routine) == 0 and inspection['INSPECTION_NAME'] == '血常规':
                try:
                    sub_inspection = inspection['sub_inspection']
                    for item in sub_inspection:
                        name = item['SUB_INSPECTION_EN']
                        result = float(item['SUB_INSPECTION_RESULT'])
                        blood_routine[name] = result
                except:
                    pass
        try:
            BASO, EO, HCT, HGB = blood_routine['BASO%'], blood_routine['EO%'], blood_routine['HCT'], blood_routine[
                'HGB']
            LYM, MCH, MCHC, MCV = blood_routine['LYM%'], blood_routine['MCH'], blood_routine['MCHC'], blood_routine[
                'MCV']
            MONO, MPV, NEUT, PCT = blood_routine['MONO%'], blood_routine['MPV'], blood_routine['NEUT%'], blood_routine[
                'PCT']
            PDW, P_LCR, PLT, RBC = blood_routine['PDW'], blood_routine['P-LCR'], blood_routine['PLT'], blood_routine[
                'RBC']
            RDW_CV, RDW_SD, WBC = blood_routine['RDW-CV'], blood_routine['RDW-SD'], blood_routine['WBC']
        except:
            BASO, EO, HCT, HGB, LYM, MCH, MCHC, MCV, MONO, MPV = '', '', '', '', '', '', '', '', '', ''
            NEUT, PCT, PDW, P_LCR, PLT, RBC, RDW_CV, RDW_SD, WBC = '', '', '', '', '', '', '', '', ''
    else:
        A_G, BUN_CREA, AST_ALT, ALB, ALP, ALT, ANION = '', '', '', '', '', '', ''
        AST, BUN, CA, CHOL, CK, Cl, CO2, CREA = '', '', '', '', '', '', '', ''
        DBIL, GGT, GLO, GLU, HDL_C, IBIL, K, LDH = '', '', '', '', '', '', '', ''
        LDL_C, Lp, MG, Na, OSM, Pi, SA, TBA = '', '', '', '', '', '', '', ''
        TBIL, TG, TP, URIC = '', '', '', ''
        BASO, EO, HCT, HGB, LYM, MCH, MCHC, MCV, MONO, MPV = '', '', '', '', '', '', '', '', '', ''
        NEUT, PCT, PDW, P_LCR, PLT, RBC, RDW_CV, RDW_SD, WBC = '', '', '', '', '', '', '', '', ''
    return A_G, BUN_CREA, AST_ALT, ALB, ALP, ALT, ANION, AST, BUN, CA, CHOL, CK, Cl, CO2, CREA, DBIL, GGT, GLO, \
           GLU, HDL_C, IBIL, K, LDH, LDL_C, Lp, MG, Na, OSM, Pi, SA, TBA, TBIL, TG, TP, URIC, BASO, EO, HCT, HGB, \
           LYM, MCH, MCHC, MCV, MONO, MPV, NEUT, PCT, PDW, P_LCR, PLT, RBC, RDW_CV, RDW_SD, WBC


# 抽取检查：心电图，X线，病理检查，CT，MR，超声
def examination(data):
    def exam_temp(item):
        text = []
        if item in data:
            for i in range(len(data[item])):
                try:
                    item_name = data[item][i]['EXAMINATION_ITEM']
                except:
                    item_name = ''
                try:
                    opinion = data[item][i]['DIAGNOSIS_OPINION']
                except:
                    opinion = ''
                try:
                    findings = data[item][i]['EXAMINATION_FINDINGS']
                except:
                    findings = ''

                try:
                    diag = data[item][i]['CLINICAL_DIAGNOSIS']
                except:
                    diag = ''
                try:
                    area = data[item][i]['EXAMINATION_AREA']
                except:
                    area = ''
                combine_text = item_name + ' ' + area + ' ' + findings + ' ' + opinion + ' ' + diag
                text.append(combine_text)
        return ' '.join(text)

    # CT，MR，ECT
    ct_report = exam_temp("ct_reports")
    mr_reports = exam_temp("mr_reports")
    ect_report = exam_temp('ect_reports')

    # 心电图
    eleccar = []
    if 'electrocardiogram_reports' in data:
        for i in range(len(data['electrocardiogram_reports'])):
            try:
                eleccar.append(data['electrocardiogram_reports'][i]['ECG_DIAGNOSTIC_OPINION'])
            except:
                pass
    elec_report = ' '.join(eleccar)

    # X线
    xray = []
    if 'xray_image_reports' in data:
        for i in range(len(data['xray_image_reports'])):
            try:
                xray.append(data['xray_image_reports'][i]["EXAMINATION_FINDINGS"])
            except:
                pass
            try:
                xray.append(data['xray_image_reports'][i]["SUGGESTION"])
            except:
                pass
            try:
                xray.append(data['xray_image_reports'][i]["CLINICAL_DIAGNOSIS"])
            except:
                pass
    xray_report = ' '.join(xray)

    # 病理检查
    pathology = []
    if 'pathology_reports' in data:
        for i in range(len(data['pathology_reports'])):
            try:
                pathology.append(data['pathology_reports'][i]['SPECIMENS_NAME'])
            except:
                pass
            try:
                pathology.append(data['pathology_reports'][i]['UNDER_MICROSCOPE'])
            except:
                pass
            try:
                pathology.append(data['pathology_reports'][i]['VISIBLE_PATHOLOGY'])
            except:
                pass
            try:
                pathology.append(data['pathology_reports'][i]['CLINICAL_DIAGNOSIS'])
            except:
                pass
            try:
                pathology.append(data['pathology_reports'][i]['PATHOLOGY_DIAGNOSIS'])
            except:
                pass
    pathology_report = ' '.join(pathology)
    # CT

    # 超声
    ultrasonic = []
    if 'ultrasonic_diagnosis_reports' in data:
        for i in range(len(data['ultrasonic_diagnosis_reports'])):
            try:
                ultrasonic.append(data['ultrasonic_diagnosis_reports'][i]['BODY_PARTS'])
            except:
                pass
            try:
                ultrasonic.append(data['ultrasonic_diagnosis_reports'][i]['"DIAGNOSIS_CONTENTS"'])
            except:
                pass
            try:
                ultrasonic.append(data['ultrasonic_diagnosis_reports'][i]['DIAGNOSIS_RESULT'])
            except:
                pass
    ult_report = ' '.join(ultrasonic)

    total_reports = ct_report + ' ' + mr_reports + ' ' + ect_report + ' ' + elec_report + ' ' + xray_report + ' ' + pathology_report + ' ' + ult_report  # 将所有检查结果合在一起

    return total_reports


# 抽取现病史：部位，症状，疾病，检查
def present_illness_history(data):
    present_illness = []
    if "admissions_records" in data:
        try:
            text = data["admissions_records"][0]["PRSENT_ILLNESS_HISTORY"]
        except:
            text = ""
    else:
        text = ""
    return text


# 抽取首次病程记录：部位，症状，疾病，检查，初步诊断
def first_course_records(data):
    if "first_course_records" in data:
        try:
            basis = data["first_course_records"][0]["DIAGNOSIS_BASIS"]
        except:
            basis = ''
        try:
            diag = data["first_course_records"][0]["PRELIMINARY_DIAGNOSIS"]
        except:
            diag = ''
        try:
            feature = data["first_course_records"][0]["MEDICAL_FEATURE"]
        except:
            feature = ''
        try:
            diff_diag = data["first_course_records"][0]["DIFFERENTIAL_DIAGNOSIS"]
        except:
            diff_diag = ''
        try:
            treat_plan = data["first_course_records"][0]["TREATMENT_PLAN"]
        except:
            treat_plan = ""
    else:
        basis = ""
        diag = ""
        feature = ''
        diff_diag = ''
        treat_plan = ""

    total_text = basis + feature + diff_diag + diag + treat_plan
    return total_text


# 抽取医嘱：用药，操作
def medical_order_extract(data):
    common_drugs = ['0.9%氯化钠注射液(非PVC)', '葡萄糖氯化钠注射液(非PVC)', '0.9%氯化钠注射液',
                    '5%葡萄糖注射液(非PVC)', '5%葡萄糖注射液']
    medical_order = []
    if "medicine_order" in data:
        try:
            for i in range(len(data["medicine_order"])):
                order = data["medicine_order"][i]["ORDER_NAME"]
                if order not in common_drugs:
                    medical_order.append(order)
        except:
            pass
    return " ".join(medical_order)


# 抽取手术记录
def operation_records(data):
    operation_name = []
    operation_record = []
    if "operation_records" in data:
        for i in range(len(data["operation_records"])):
            try:
                operation_name.append(data["operation_records"][i]["OPERATION"])
            except:
                pass
            try:
                operation_record.append(data["operation_records"][i]["OPERATION_STEPS"])
            except:
                pass
    return ' '.join(operation_name), ' '.join(operation_record)


def course_record(data):
    all_records = []
    if "course_records" in data:
        records = data["course_records"]
        for i in range(len(records)):
            try:
                all_records.append(records[i]["WARD_INSPECTION_RECORD"])
            except:
                pass
    return ' '.join(all_records)


def discharge_record(data):
    if "discharge_records" in data:
        if "HOSPITAL_DISCHARGE_ORDER" in data["discharge_records"][0]:
            discharge_order = data["discharge_records"][0]["HOSPITAL_DISCHARGE_ORDER"]
        else:
            discharge_order = ''
        if "TREATMENT_COURSE" in data["discharge_records"][0]:
            treat_course = data["discharge_records"][0]["TREATMENT_COURSE"]
        else:
            treat_course = ''
    else:
        discharge_order = ''
        treat_course = ''
    return treat_course + ' ' + discharge_order


# 抽取出院诊断和入院诊断
def adm_dis_diagnose(data):
    if "discharge_records" in data:
        if "HOSPITAL_DISCHARGE_DIAGNOSE" in data["discharge_records"][0]:
            diag = data["discharge_records"][0]["HOSPITAL_DISCHARGE_DIAGNOSE"]
            # 去掉出院诊断中的无关内容，标点，特殊符号，编号，首末空格
            pro_diag = re.sub('.+?诊断[：]*', "", diag)
            pro_diag = re.sub('[(（].+?[）)]', "", pro_diag)
            pro_diag = re.sub('[\t\r\n]', "", re.sub('[、，。？?,]', ' ', pro_diag))
            discharge_diags = re.sub('\d[\.、 ]', '', pro_diag).strip()
        else:
            discharge_diags = ""

        if "HOSPITAL_ADMISSION_DIAGNOSE" in data["discharge_records"][0]:
            diag = data["discharge_records"][0]["HOSPITAL_ADMISSION_DIAGNOSE"]
            pro_diag = re.sub('.+?诊断[：]*', "", diag)
            pro_diag = re.sub('[(（].+?[）)]', "", pro_diag)
            pro_diag = re.sub('[\t\r\n]', "", re.sub('[、，。？?,]', ' ', pro_diag))
            admission_diags = re.sub('\d[\.、 ]*', '', pro_diag).strip()
        else:
            admission_diags = ""
    else:
        discharge_diags = ""
        admission_diags = ""
    return admission_diags, discharge_diags


# 抽取入院科室
def admission_dept(data):
    if "visit_info" in data:
        if "ADMISSION_DEPT" in data["visit_info"][0]:
            dept = data["visit_info"][0]["ADMISSION_DEPT"]
        else:
            dept = ''
    else:
        dept = ''
    return dept


# 抽取标签：病案首页主诊断的ICD编码
def main_diagnose(data):
    if 'medical_record_home_page' in data:
        if 'dis_main_diag' in data['medical_record_home_page'][0]:
            try:
                main_diag_code = data['medical_record_home_page'][0]['dis_main_diag'][0]['DIS_DIAG_CODE']
            except:
                main_diag_code = ""
        else:
            main_diag_code = ""
    else:
        main_diag_code = ""
    return main_diag_code


t1 = time.time()
# raw_data = []
sum_text = []

with open('/home/pkudata/medical_home_page_source_data/medical_home_page_8.2M.data', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        # dept = admission_dept(data)
        # age, gender, marital_status = personal_information(data)
        chief_complaint = Chief_Complaint(data)
        # temperature, pulse, breath, HBP, LBP = vital_signs(data)
        # A_G, BUN_CREA, AST_ALT, ALB, ALP, ALT, ANION, AST, BUN, CA, CHOL, CK, Cl, CO2, CREA, DBIL, GGT, GLO, \
        # GLU, HDL_C, IBIL, K, LDH, LDL_C, Lp, MG, Na, OSM, Pi, SA, TBA, TBIL, TG, TP, URIC, BASO, EO, HCT, HGB, \
        # LYM, MCH, MCHC, MCV, MONO, MPV, NEUT, PCT, PDW, P_LCR, PLT, RBC, RDW_CV, RDW_SD, WBC = inspection_reports(data)
        exam = examination(data)
        present_illness = present_illness_history(data)
        first_records = first_course_records(data)
        orders = medical_order_extract(data)
        op_name, op_record = operation_records(data)
        cour_record = course_record(data)
        dis_record = discharge_record(data)
        admission_diags, discharge_diags = adm_dis_diagnose(data)
        main_diag_code = main_diagnose(data)

        text = [chief_complaint, exam, present_illness, first_records, orders, op_name, op_record, cour_record,
                dis_record, admission_diags, discharge_diags]
        sum_text.append(['\n\n'.join(text), main_diag_code])
'''
        row = [dept, age, gender, marital_status, chief_complaint, temperature, pulse, breath, HBP, LBP,
               A_G, BUN_CREA, AST_ALT, ALB, ALP, ALT, ANION, AST, BUN, CA, CHOL, CK, Cl, CO2, CREA, DBIL,
               GGT, GLO, GLU, HDL_C, IBIL, K, LDH, LDL_C, Lp, MG, Na, OSM, Pi, SA, TBA, TBIL, TG, TP, URIC,
               BASO, EO, HCT, HGB, LYM, MCH, MCHC, MCV, MONO, MPV, NEUT, PCT, PDW, P_LCR, PLT, RBC, RDW_CV,
               RDW_SD, WBC, exam, present_illness, first_records,orders, op_name, op_record,cour_record,
               dis_record, admission_diags, discharge_diags, main_diag_code]
        raw_data.append(row)

with open('test_data.csv', 'w+', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for row in raw_data:
        writer.writerow(row)
'''

with open('/home/yanrui/ICD/text_data.csv', 'w+', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for row in sum_text:
        writer.writerow(row)
t2 = time.time()
print(t2 - t1)