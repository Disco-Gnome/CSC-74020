# CSC 74020 Course Material (excl. Lecture Slides)

The purpose of this repo is to store the notebooks we cover in class as well as template code for certain assignments. This repo is NOT self contained and the material in the notebooks often refer to slides that are not posted here.

## Features
- Review of Python basics (Week0)
- Notebooks used or referenced in weekly lectures
- Assignment 4 PDF chatbot template

In addition to these notebooks, slides, and material discussed in class, please refer to the reading materials cited in the course schedule from the following: 
- Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems, 3nd Edition, by Aurélien Géron. Great resource for additional notebook on [Github](https://github.com/ageron/handson-ml3)  
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](http://web.stanford.edu/~hastie/ElemStatLearn/) by Trevor Hastie, Robert Tibshirani, Jerome Friedman
- Pattern Recognition and Machine Learning by [Christopher Bishop at Microsoft Research](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
- [Machine Learning: a Probabilistic Perspective](https://probml.github.io/pml-book/book0.html) and [link 2](http://noiselab.ucsd.edu/ECE228/Murphy_Machine_Learning.pdf) by Kevin P. Murphy
- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/first/ebook/the-book.html)  by Richard S. Sutton and Andrew G. Barto 

Additional Resources:
- Learning From Data, Yaser S. Abu-Mostafa, Malik Magdon-Ismail, Hsuan-Tien Lin: ISBN: 9781600490064. [Supporting Material](http://amlbook.com/support.html)
- [Deep Learning](https://www.deeplearningbook.org/) by Ian Goodfellow, Yoshua Bengio, and Aaron Courville 


## Setup

### Prerequisites

You need to have Python 3.8 or later to use this application. If you don't have Python installed, you can download it from the official site: https://www.python.org/downloads/

### Steps

1. Clone the repository to your local machine.

   ```bash
   git clone https://github.com/Markovian99/CSC-74020.git
   ```
   
2. Navigate to the cloned project directory.

   ```bash
   cd CSC-74020
   ```
   
3. Install the necessary packages using pip. (It is recommended to use a virtual environment)

   ```bash
   pip install -r requirements.txt
   ```
   
4. For Assignment 4, in the project's root directory, create a `.env` file to store your API keys securely.
   
5. Open the `.env` file using any text editor and enter your API keys as shown below (if using OpenAI):

   ```bash
   export OPENAI_API_KEY = "YOUR OPENAI API KEY"
   ```
   If using Azure for OPEN AI, also include
   ```bash
   export OPENAI_API_BASE = "YOUR OPENAI API BASE"
   export OPENAI_API_TYPE = "YOUR OPENAI API TYPE"
   export OPENAI_API_VERSION = "YOUR OPENAI API VERSION"
   ```

## Usage

To run the PDFLucy application:

```bash
cd src
streamlit run app.py
```

## License

This project is licensed under the terms of the Apache License 2.0. For more details, please see the [LICENSE](LICENSE) file.

## Support

For any questions or issues, please contact the maintainers, or raise an issue in the GitHub repository.
