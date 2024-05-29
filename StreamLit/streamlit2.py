import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your LLaMA model and tokenizer
model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=200,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    st.title("LLaMA Model Inference")

    prompt = st.text_area("Enter your prompt", height=200)

    if st.button("Generate Response"):
        with st.spinner("Generating response..."):
            response = generate_response(prompt)
        st.success(response)

if __name__ == "__main__":
    main()