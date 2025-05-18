import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib

models = {
    "Logistic Regression": joblib.load("logistic_model.pkl"),
    "SVM": joblib.load("svm_model.pkl"),
    "KNN": joblib.load("knn_model.pkl"),
    "Decision Tree": joblib.load("dt_model.pkl")
}

feature_names = ['age', 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


class HeartDiseaseApp:
    def __init__(self, root):
        self.root = root
        
        root.title("Heart Disease Prediction")
        root.geometry("600x750")

        tk.Label(root, text="Heart Disease Prediction", font=("inter", 24, "bold"), fg='#588DFF').pack(pady=10)
        self.style = ttk.Style()
        self.inputs = {}

        # Age
        self.add_entry("age", "Age")

        # Sex (0 = female, 1 = male)
        self.add_radio_group("sex", "Sex", {"Female": 0, "Male": 1})

        # Chest pain type (0–3)
        self.add_dropdown("cp", "Chest Pain Type", {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-anginal Pain": 2,
            "Asymptomatic": 3
        })

        # Max heart rate
        self.add_entry("thalach", "Max Heart Rate Achieved")

        # Exercise-induced angina
        self.add_dropdown("exang", "Exercise-Induced Angina", {"No": 0, "Yes": 1})

        # Oldpeak
        self.add_entry("oldpeak", "ST Depression (Oldpeak)")

        # Slope of ST segment
        self.add_dropdown("slope", "Slope of Peak Exercise ST", {
            "Upsloping": 0,
            "Flat": 1,
            "Downsloping": 2
        })

        # Number of major vessels
        self.add_dropdown("ca", "Number of Vessels", {str(i): i for i in range(4)})

        # Thalassemia
        self.add_dropdown("thal", "Thalassemia", {
            "Normal": 1,
            "Fixed Defect": 2,
            "Reversible Defect": 3
        })

        # Model selection
        tk.Label(root, text="Select Model").pack()
        self.model_choice = ttk.Combobox(root, values=list(models.keys()), state="readonly")
        self.model_choice.current(0)
        self.model_choice.pack(pady=5)


        # Predict button
        ttk.Button(root, text="Predict", command=self.predict).pack(pady=20)

        # Output
        self.result_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.result_label.pack()

    def add_entry(self, key, label):
        frame = tk.Frame(self.root)
        frame.pack(pady=5)
        tk.Label(frame, text=label + ": ",font=("inter", 12, "bold"), fg='#383838', width=30,anchor='w').pack(side="left")
        entry = ttk.Entry(frame) 
        entry.pack(side="right", ipady=5)
        self.inputs[key] = entry

    def add_dropdown(self, key, label, options: dict):
        frame = tk.Frame(self.root)
        frame.pack(pady=5)
        tk.Label(frame, text=label + ": ",font=("inter", 12, "bold"), fg='#383838', width=30, anchor="w").pack(side="left")
        combo = ttk.Combobox(frame, values=list(options.keys()), state="readonly", width=16)
        combo.pack(side="left", ipady=5)
        combo.current(0)
        self.inputs[key] = (combo, options)
        

    def add_radio_group(self, key, label, options: dict):
        frame = tk.Frame(self.root)
        frame.pack(pady=5)

        tk.Label(frame, text=label + ":", font=("inter", 12, "bold"), fg='#383838', anchor="w", width=24).pack(side="left")

        var = tk.StringVar()
        var.set(next(iter(options.keys())))  # Set first option as default

        for text, value in options.items():
            rb = ttk.Radiobutton(frame, text=text, variable=var, value=text)
            rb.configure()
            rb.pack(anchor="w", padx=20, side=tk.LEFT)

        self.inputs[key] = (var, options)
        
    def predict(self):
        try:
            input_data = []
            for key, widget in self.inputs.items():
                if isinstance(widget, tuple):  # dropdown
                    label, mapping = widget
                    selected = label.get()
                    value = mapping[selected]
                else:  # entry
                    value = float(widget.get())
                input_data.append(value)

            model = models[self.model_choice.get()]
            result = model.predict([input_data])[0]
            confidence = model.predict_proba([input_data])[0][int(result)] if hasattr(model, 'predict_proba') else None

            if result == 1:
                msg = "⚠️ High Risk of Heart Disease"
                color = "red"
            else:
                msg = "🟢 Low Risk of Heart Disease"
                color = "green"

            if confidence is not None:
                msg += f"\nConfidence: {confidence*100:.2f}%"

            self.result_label.config(text=msg, fg=color)
        except Exception as e:
            messagebox.showerror("Error", f"Something went wrong:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = HeartDiseaseApp(root)
    root.mainloop()