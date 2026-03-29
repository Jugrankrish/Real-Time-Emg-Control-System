🧠 Overview

This project presents a real-time EMG-based bio-control system that converts muscle activity into control signals. It uses surface electrodes to capture EMG signals, which are amplified using an AD620 instrumentation amplifier and processed by an ESP32. The system focuses on building a complete pipeline from signal acquisition to control output on embedded hardware, rather than relying only on offline analysis.

⚙️ System Design

The hardware setup includes electrodes, an AD620 amplifier for signal conditioning, and an ESP32 for analog-to-digital conversion and processing. The EMG signals are filtered and normalized before being used for further analysis. This design ensures low-cost implementation while maintaining real-time performance suitable for practical applications.

🔬 Methodology

After acquiring the EMG signal, features such as RMS, mean absolute value, and variance are extracted to represent muscle activity. Two approaches were explored: unsupervised learning using KMeans clustering and supervised learning using models like Logistic Regression and SVM. Structured datasets with rest, weak, and strong contractions were used to evaluate classification performance.

📊 Results & Key Findings

Experimental results showed that KMeans clustering produced high silhouette scores, especially during rest conditions. However, this revealed that clustering primarily captures signal patterns or time-based regimes rather than true muscle intensity levels. This insight highlighted the limitation of unsupervised learning for EMG classification and emphasized the importance of labeled datasets.

🚀 Conclusion & Applications

The project demonstrates that while EMG signals can be reliably acquired and processed in real time, accurate classification of muscle activity requires supervised learning and proper dataset design. This system provides a strong foundation for applications such as prosthetic control, human-computer interaction, and assistive technologies, with scope for future improvements using multi-channel EMG and advanced machine learning models.
