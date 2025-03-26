# 🚀 Project Name

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Team](#team)

---

## 🎯 Introduction
Data profiling evaluates data based on factors such as data accuracy, consistency and timeliness to show if the data is lacking consistency or accuracy.

## 🎥 Demo
Video of the Demo is attached in the "https://github.com/ewfx/gaidp-deepcore/tree/main/artifacts/demo"

## 💡 Inspiration
Part of the Hackathon

## ⚙️ What It Does
When prompted in the UI, the LLM reads the data profiling rules, checks the data and reports the anomalies. 

## 🛠️ How We Built It
This is built on Langchain Agent with Groq LLama Model with unsupersvised machine learning techniques to indetify anomalies.

## 🚧 Challenges We Faced
Doing this on a personal laptop managing both office and personal time.
E2E implementation as per our design is not implemented as per the hackathon instructions and still there is a lot scope to improve our code.

## 🏃 How to Run
1. Clone the repository  
   ```sh
   git clone https://github.com/ewfx/gaidp-deepcore.git
   ```
Run the automated_data_profiling.py in pycharm/IDE that supports python
In the terminal please paste this command "streamlit run .\automated_data_profiling.py" and hit enter

   ```

## 🏗️ Tech Stack
pip install streamlit 
pip install pandas
pip install -U scikit-learn
pip install langchain_groq 
pip install io
pip install langchain
pip install langchain_experimental

## 👥 Team
Rathiesh Anarajula
Naresh Dindi
Siva Krishna Chaitanya R
Sai Gubba
