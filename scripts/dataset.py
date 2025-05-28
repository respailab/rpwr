import pandas as pd
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

filtered_data_diagnosed = pd.read_csv("data/preprocessed_data_350.csv")

fibromyalgia_df = filtered_data_diagnosed[filtered_data_diagnosed['final-diagnosis'] == 'fibromyalgia']
rheumatoid_arthritis_df = filtered_data_diagnosed[filtered_data_diagnosed['final-diagnosis'] == 'rheumatoid arthritis']

f_kb = int((len(fibromyalgia_df)*80)/100)
r_kb = int((len(rheumatoid_arthritis_df)*80)/100)

print(f"Fibromyalgia Knowledge Base Size: {f_kb}")
print(f"Rheumatoid Arthritis Knowledge Base Size: {r_kb}")

fibromyalgia_knowledge_base = fibromyalgia_df.iloc[:f_kb]
rheumatoid_arthritis_knowledge_base = rheumatoid_arthritis_df.iloc[:r_kb]

knowledge_base_df = pd.concat([fibromyalgia_knowledge_base, rheumatoid_arthritis_knowledge_base], ignore_index=True)

fibromyalgia_test = fibromyalgia_df.iloc[f_kb:]
rheumatoid_arthritis_test = rheumatoid_arthritis_df.iloc[r_kb:]

test_df = pd.concat([fibromyalgia_test, rheumatoid_arthritis_test], ignore_index=True)
# test_df.to_csv("data/test_data.csv", index=False)
print(f"Fibromyalgia Test Size: {len(fibromyalgia_test)}")
print(f"Rheumatoid Arthritis Test Size: {len(rheumatoid_arthritis_test)}")

def create_patient_prompt(patient_data):
    prompt = f"""
Patient Information:

Age: {patient_data['age']}
Gender: {patient_data['gender']}
Problems: {patient_data['problems']}
Duration of Problem: {patient_data['total_days']} days
Pain Location: {patient_data['pain-here']}
Pain Areas (Image 1): {patient_data['image1-loc-mapped']}
Pain Areas (Image 2): {patient_data['image2-loc-mapped']}
Joint Deformity: {patient_data['joint-deformit']}
Swelling Areas (Image 1): {patient_data['swelling1-mapped']}
Swelling Areas (Image 2): {patient_data['swelling2-mapped']}
Swelling Pattern: {patient_data['swelling-constant']}
Redness in Swollen Joints: {patient_data['red-swollen']}
Warm Joints: {patient_data['warm-joints']}
Pain Time: {patient_data['pain-time']}
Sleep Disturbance: {patient_data['sleep-disturb']}
Sleep Hours: {patient_data['sleep-hours']}
Physical Activity Impact: {patient_data['increase-pain']}
Rest Impact: {patient_data['rest-increase']}
Painkillers: {patient_data['painkillers']}
Painkiller Response: {patient_data['response-painkillers']}
"""

    # Handle missing values
    
    # Handle missing values
    if pd.notnull(patient_data['other-symptoms']):
        if patient_data['other-symptoms'] == 'No':
            prompt += f"Other symptoms: No\n"
        else:
            prompt += f""
            
    if pd.notnull(patient_data['temp']):
        if patient_data['temp'] == 'no':
            prompt += f"Fever: No\n"
        else:
            prompt += f"Fever: {patient_data['temp']}\n"

    if pd.notnull(patient_data['skin-rash']):
        if patient_data['skin-rash'] == 'no':
            prompt += f"Skin Rash: No\n"
        else:
            prompt += f"Skin Rash: {patient_data['skin-rash']}\n"

    if pd.notnull(patient_data['rash-sun']):
        if patient_data['rash-sun'] == 'no':
            prompt += f"Rash in Sun: No\n"
        else:
            prompt += f"Rash in Sun: {patient_data['rash-sun']}\n"

    if pd.notnull(patient_data['mouth-ulcers']):
        if patient_data['mouth-ulcers'] == 'no':
            prompt += f"Mouth Ulcers: No\n"
        else:
            prompt += f"Mouth Ulcers: {patient_data['mouth-ulcers']}\n"

    if pd.notnull(patient_data['eye-grittiness']):
        if patient_data['eye-grittiness'] == 'no':
            prompt += f"Eye Grittiness: No\n"
        else:
            prompt += f"Eye Grittiness: {patient_data['eye-grittiness']}\n"

    if pd.notnull(patient_data['eye-drops']):
        if patient_data['eye-drops'] == 'no':
            prompt += f"Eye Drops: No\n"
        else:
            prompt += f"Eye Drops: {patient_data['eye-drops']}\n"

    if pd.notnull(patient_data['difficult-swallowing']):
        if patient_data['difficult-swallowing'] == 'no':
            prompt += f"Swallowing Difficulty: No\n"
        else:
            prompt += f"Swallowing Difficulty: {patient_data['difficult-swallowing']}\n"

    if pd.notnull(patient_data['difficult-wakingup']):
        if patient_data['difficult-wakingup'] == 'no':
            prompt += f"Difficulty Waking Up: No\n"
        else:
            prompt += f"Difficulty Waking Up: {patient_data['difficult-wakingup']}\n"

    if pd.notnull(patient_data['difficult-sitting']):
        if patient_data['difficult-sitting'] == 'no':
            prompt += f"Difficulty Sitting Up: No\n"
        else:
            prompt += f"Difficulty Sitting Up: {patient_data['difficult-sitting']}\n"

    if pd.notnull(patient_data['psoriasis']):
        if patient_data['psoriasis'] == 'no':
            prompt += f"Psoriasis: No\n"
        else:
            prompt += f"Psoriasis: {patient_data['psoriasis']}\n"

    if pd.notnull(patient_data['longer-heal-time']):
        if patient_data['longer-heal-time'] == 'no':
            prompt += f"Healing Time: No\n"
        else:
            prompt += f"Healing Time: {patient_data['longer-heal-time']}\n"

    if pd.notnull(patient_data['painful-eye']):
        if patient_data['painful-eye'] == 'no':
            prompt += f"Painful Eye: No\n"
        else:
            prompt += f"Painful Eye: {patient_data['painful-eye']}\n"

    if pd.notnull(patient_data['cough']):
        if patient_data['cough'] == 'no':
            prompt += f"Cough: No\n"
        else:
            prompt += f"Cough: {patient_data['cough']}\n"

    if pd.notnull(patient_data['previous-ra']):
        if patient_data['previous-ra'] == 'no':
            prompt += f"Previous History of Joint Pain, Swelling: No\n"
        else:
            prompt += f"Previous History of Joint Pain, Swelling: Yes\n"

    if pd.notnull(patient_data['medicines']):
        if patient_data['medicines'] == 'no':
            prompt += f"Medication History: No\n"
        else:
            prompt += f"Medication History: {patient_data['medicines']}\n"

    if pd.notnull(patient_data['medicines-now']):
        if patient_data['medicines-now'] == 'no':
            prompt += f"Current Medications: No\n"
        else:
            prompt += f"Current Medications: {patient_data['medicines-now']}\n"

    if pd.notnull(patient_data['final-diagnosis']):
        if patient_data['final-diagnosis'] == "fibromyalgia":
            patient_data['final-diagnosis'] = "Not Rheumatoid Arhtritis"
        else:
            patient_data['final-diagnosis'] = "Rheumatoid Arhtritis"
        prompt += f"Final Diagnosis: {patient_data['final-diagnosis']}\n"

    return prompt

knowledge_base_docs = []
for index, row in knowledge_base_df.iterrows():
    patient_prompt = create_patient_prompt(row)
    knowledge_base_docs.append(patient_prompt)
    

openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

db = Chroma.from_texts(knowledge_base_docs, embedding=embeddings, persist_directory="knowledge_base")
retriever = db.as_retriever()

actual_diagnosis = []
test_docs = []

for index, row in test_df.iterrows():
    patient_prompt = create_patient_prompt(row)
    actual_diagnosis.append(patient_prompt.split("\n")[-2].split(": ")[-1])
    patient_prompt = ("\n").join(patient_prompt.split("\n")[3:-2])
    test_docs.append(patient_prompt)


test_set = pd.DataFrame({
    "Patient Prompt": test_docs,
    "Actual Diagnosis": actual_diagnosis
})

test_set.to_csv("data/test_data.csv", index=False)
print(f"\nTotal knowledge base size: {len(knowledge_base_docs)} and Total test Set Size: {len(test_set)}")