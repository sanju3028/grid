import os
import pandas as pd
import spacy
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize NLP for rule extraction
nlp = spacy.load("en_core_web_sm")

# Initialize LLM for generating insights
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load compliance rules from a file (simplified example)
def load_rules(rules_file):
    with open(rules_file, "r") as file:
        rules = file.readlines()
    return [rule.strip() for rule in rules]

# Load logs from different formats (simplified example)
def parse_logs(log_file):
    if log_file.endswith(".csv"):
        logs_df = pd.read_csv(log_file)
    elif log_file.endswith(".txt"):
        with open(log_file, "r") as file:
            logs = file.readlines()
        logs_df = pd.DataFrame({"log_text": logs})
    # Add handling for other formats like PDF if needed
    return logs_df

# Extract and infer rules using SpaCy (simplified example)
def extract_rules(compliance_rules):
    extracted_rules = []
    for rule in compliance_rules:
        doc = nlp(rule)
        for sent in doc.sents:
            extracted_rules.append(sent.text)
    return extracted_rules

# Detect compliance breaches in logs
def detect_breaches(extracted_rules, logs_df):
    breaches = []
    for index, row in logs_df.iterrows():
        log_text = row["log_text"]
        for rule in extracted_rules:
            if rule in log_text:
                breaches.append((rule, log_text))
    return breaches

# Generate insights using LLM
def generate_insights(breach_description):
    prompt = f"Breach description: {breach_description}. Remediation: "
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = model.generate(input_ids, max_length=100, num_return_sequences=1)
    insights = tokenizer.decode(generated[0], skip_special_tokens=True)
    return insights

# Main function
def main():
    rules = load_rules("compliance_rules.txt")
    logs_df = parse_logs("logs.csv")

    extracted_rules = extract_rules(rules)
    breaches = detect_breaches(extracted_rules, logs_df)

    for rule, log in breaches:
        insights = generate_insights(rule)
        print("Compliance breach:", rule)
        print("Log:", log)
        print("Actionable insights:", insights)
        print("-" * 50)

if __name__ == "__main__":
    main()
