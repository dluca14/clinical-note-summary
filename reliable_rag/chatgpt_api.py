# %%
import math

import openai
import pandas as pd
import os
from docx import Document  # For creating Word files


def setup_openai_client():
    openai.api_key = ""


def generate_primaries_detailed(code_list, free_text):
    text = f"""I need you to create a thorough Clinical Validation and Diagnosis Analysis, analyze carefully the attached clinician note and then the Given the list of medical conditions and the procedures reported in the claim list:
            {code_list}
            I want you to be very detailed and use the text to ensure that there is sufficient clinical evidence to support the presence of the medical conditions and procedures.
            Here are the clinician's notes: {free_text}
            make sure that you extract the clinical information accurately from the files and analyze the attachments and compare them to each diagnosis code to ensure the clinical evidence and accuracy of the reported medical conditions.
            I want you to be very detailed, provide with the evidence and use the text to ensure that there is the necessary clinical evidence to support the presence of the medical conditions and procedures. For example, to proof the presence of sepsis the patient needs to meet at least 2 or more sepsis criteria (for example, T or temperature> 38 or T or temperature< 36 or Heart Rate>90, etc.)
            I need answers in the format below, quoting from the provided clinician notes to support your decisions on whether there is evidence for a medical condition or not:
            Patient Information:
            •	Age: 
            •	Gender: 
            •	Length of Stay: 
                            •	Patient age
                            •	Past Medical History
                            •	Presenting problems/complaints
                            •	Vitals, if appropriate to the project
                            •	Pertinent labs, testing and results
                            •	Pertinent treatment
                            •	Documentation to support the decision based on clinical validation reference
                            •	Rationale of: Based on this information, the patient does not meet diagnostic criteria for
            • Medications: 
            • Section Reference: 
            • Quoted Evidence: 
            • Rationale: 
            • Conclusion:
            If any of the medical conditions or procedures is not supported, mention 'NOT SUPPORTED' next to it.
            You must not infer or fabricate any information. If a detail is absent from the clinical notes, it cannot be considered supported. Additionally, if a procedure listed in the documentation did not occur at the hospital where the patient was treated, it should be flagged as unsupported.
            If any of the medical conditions or procedures is not supported, mention 'NOT SUPPORTED' next to it.

            You have to always follow the same format, you cannot be lazy. 
            For example:

            Atherosclerosis of Native Arteries of Extremities with Gangrene, Bilateral 
            Feet and Toes (I70.263)
            • Past Medical History: The patient has a history of chronic heart failure (CHF) and CKD stage 
            II. Bilateral foot and toe necrosis and gangrene have been progressively worsening.
            • Presenting Complaints: The patient presented with bilateral foot pain and worsening 
            necrosis.
            • Findings: Gangrene in both feet confirmed by physical exam and patient history.
            • Lab Results:
                o WBC Count: 5.8 K/mm3 (within normal range).
                o CRP: Elevated at 112 mg/L, indicating significant inflammation and infection.
            • Vital Signs:
                o Temperature: 97.7°F (oral), normal.
                o Blood Pressure: 113/88 mmHg, stable.
                o Pulse Rate: 56 bpm, normal.
            • Medications: Cefepime, vancomycin, and acetaminophen were administered.
            • Section Reference: Documented under "Chief Complaint" and "History of Present Illness."
            • Quoted Evidence: "The patient presented today due to bilateral foot pain and worsening 
            necrosis" and "Patient mentioned that he had issues with his foot for the past 2 years... He 
            already went amputation of right foot first toe, left foot first and fifth toe."
            • Rationale: The diagnosis is supported by documented foot and toe necrosis and gangrene, 
            with elevated CRP suggesting ongoing inflammation.

            Sepsis has 2 criteria in order to be validated as sepsis:
            SIRS Criteria (≥2 meets SIRS definition)
            Temp >38°C (100.4°F) or <36°C (96.8°F)
            Heart rate >90.
            Respiratory rate >20 or PaCO₂ <32 mm Hg.
            WBC >12,000/mm³, <4,000/mm³, or >10%' bands.
            Sepsis Criteria (SIRS + Source of Infection):
            Suspected or present source of infection.

            Independent measurements such as lab tests, vitals, and imaging must take precedence. If these objective data points contradict the clinical documentation, this discrepancy should be highlighted. While medication effects and clinician notes are relevant, all factors must be explicitly mentioned without exception.

            You must not infer or fabricate any information. If a detail is absent from the clinical notes, it cannot be considered supported. Additionally, if a procedure listed in the documentation did not occur at the hospital where the patient was treated, it should be flagged as unsupported.

            The following terms may suggest nicotine dependence:

            Cigarettes per day (CPD)
            Pack years
            Pack
            Heavy smoker
            Light smoker
            Smoking frequency
            Chain smoking
            First cigarette after waking
            Quit attempts
            Relapse
            Nicotine content
            Tobacco use disorder
            Smoking cessation
            Nicotine replacement therapy (NRT)
            For substance use or stimulant abuse, the following words and drugs may indicate usage: Terms:

            Amphetamines
            Craving
            Tolerance
            Withdrawal
            Euphoria
            Agitation
            Hyperactivity
            Dependence
            Addiction
            Relapse
            Bingeing
            Overstimulation
            Insomnia
            Paranoia
            Drugs:

            Methamphetamine
            Adderall (Amphetamine/Dextroamphetamine)
            Ritalin (Methylphenidate)
            Dexedrine (Dextroamphetamine)
            Cocaine
            MDMA (Ecstasy, Molly)
            Ephedrine
            Khat
            Modafinil
            """

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": """You're a professional doctor and an expert medical coder at CMS, so you know the coding 
             guidelines by heart. Given the list of medical conditions and the procedures reported in the claim list, 
             analyze the text and compare it to each diagnosis code to ensure the clinical evidence and accuracy of the 
             reported medical conditions.
             I want you to be very detailed and use the text to ensure that there is the necessary clinical evidence to 
             support the presence of the medical conditions and procedures. 
             I need answers in the format below, quoting from the provided clinician notes to support your decisions on 
             whether there is evidence for a medical condition or not:
                Patient Information:
                •	Age: 
                •	Gender: 
                •	Length of Stay: 
                                •	Patient age
                                •	Past Medical History
                                •	Presenting problems/complaints
                                •	Vitals, if appropriate to the project
                                •	Pertinent labs, testing and results
                                •	Pertinent treatment
                                •	Documentation to support the decision based on clinical validation reference
                                •	Rationale of: Based on this information, the patient does not meet diagnostic criteria for
                • Medications: 
                • Section Reference: 
                • Quoted Evidence: 
                • Rationale: 
                • Conclusion:
                """},
            {"role": "user", "content": text}
        ],
        temperature=0.8,
        max_tokens=4096
    )

    return response.choices[0].message.content


def save_to_word(claim_nbr, result):
    # Create a new Word document
    doc = Document()

    # Add the result text to the document
    doc.add_paragraph(result)

    # Define the output path and save the document
    output_path = f'../data/test_4658/MRR_{claim_nbr}_inter_codes.docx'
    doc.save(output_path)

    print(f"Result saved to {output_path}")


def process_code_list_and_free_text(file_path):
    # Load the code list CSV file
    df = pd.read_csv(file_path)

    # Group by genereated_claim_nbr to process each claim individually
    for claim_nbr, group in df.groupby('genereated_claim_nbr'):
        # Convert the grouped rows to the required format
        code_list = group.to_string(index=False, header=False)

        # Construct the free_text file path based on claim_nbr
        free_text_file = f'../data/test_4658/MR for {claim_nbr}_extracted_150_cleaned.txt'

        if os.path.exists(free_text_file):
            with open(free_text_file, 'r', encoding='utf-8') as f:
                free_text = f.read()
        else:
            print(f"Free text file for claim {claim_nbr} not found.")
            free_text = ""

        code_list = code_list.split('\n')
        n = 10
        result = ''
        for _ in range(math.ceil(len(code_list) / 10)):
            # Call the OpenAI model for each claim
            result += generate_primaries_detailed(code_list[n - 10:n], free_text) + '\n'
            n += 10

        # Save the result to a Word document
        save_to_word(claim_nbr, result)

        # You can process or save the result as needed
        print(f"Result for {claim_nbr} were generated")


# Initialize OpenAI client
setup_openai_client()

# Process the code list and corresponding free_text files
process_code_list_and_free_text('../data/context/codes/code_list.csv')
