# Bob_Dylan_generator

## Purpose
Our project aims to collect the poetry data of Bob Dylan from kaggle(https://www.kaggle.com/code/cloudy17/bob-dylan-songs-1961-2020/input) and fine-tune multiple LLMs models to enable them to generate poetry in a style similar to Bob Dylan's. In addition to this, we have also implemented RAG (Retrieval-Augmented Generation) and evaluated and compared the outputs of these models. What's more, we provide a website user interface for testing.


## Features
- Collected a comprehensive dataset of Bob Dylan's poetry and song lyrics from various authenticated sources.
- Fine-tuned multiple Large Language Models (LLMs) to capture the essence of Bob Dylan's lyrical and poetic style.
- Implemented Retrieval-Augmented Generation (RAG) to enhance the creative output of the models by leveraging a database of Dylan's works.
- Developed a user-friendly graphical user interface (GUI) that allows users to input prompts and receive poetry generated in Bob Dylan's style. In addition, user can choose which model they want to use.
- Supported the generation of content in various formats, including song lyrics, short poems, and long-form poetry, to cater to diverse user preferences.
- Provided extensive documentation and codebase for the models' training and fine-tuning processes, offering insights for enthusiasts and researchers interested in natural language generation and stylistic emulation.
- Conducted a thorough evaluation and comparison of model outputs using a third LLM model, ensuring the quality and authenticity of the generated content.
- Enabled community contributions by allowing users to submit their own Dylan-esque poetry, which further trains and refines the models.


## Prepare
### Environment Requirements
- Python 3.8+
- Required dependencies (see requirements.txt)

### Installation Steps
After you fork and git clone the project, You should do the following steps:
1. Prepare for the virtual environment `python -m venv venv`
2. Activate virtual environment.<br/> Windows:`venv\Scripts\activate`, MacOS or Linux:`source venv/bin/activate`
3. Install required packages `pip install -r requirements.txt`

## Stages

### Data Collection and Preprocessing

1. Use a web scraper script based on `selenium` to collect artwork data from the NGA website (https://www.nga.gov/) and save it in an appropriate format, such as CSV. 
<!-- <img src="./img/image.png" alt="Description of your image" width=“400” height="300"> -->
![NGA Website](./img/image.png)

Due to the reason that NGA uses `JavaScript` and `Ajax` to generate content, using the `http.request` library will only retrieve the initial static HTML content and won't capture dynamically generated data. `Selenium`, by simulating user interactions with a browser, can load and execute JavaScript to retrieve the complete page content. Therefore, we get these images one by one using selenium.

<!-- <img src="./img/image2.png" alt="Description of your image" width=“200” height="200"> -->
![Alt text](./img/image2.png)


2. Preprocess the scraped data, including image processing and data cleaning. Ensure that the images in the dataset align with their corresponding year labels.

    2.1.  Firstly, we got the csv file that includes header columns of title, years, link. 

    <!-- <img src="./img/image3.png" alt="Description of your image" width=“200” height="100"> -->
    ![Alt text](./img/image3.png)

    2.2 Clean them and got the corresponding label(year) with local image files'name

    2.3 Fetch the images and stored it into different label folders.
    <!-- <img src="./img/folders.jpg" alt="Description of your image" width=“200” height="100"> -->
    ![Alt text](./img/folders.jpg)

### Data Augmentation
Considering the unbalanced dataset, we adopt the **offline augmentation** method to enlarge the dataset. This method is suitable for smaller datasets. You will eventually increase the dataset by a certain multiple, which is equal to the number of conversions you make. For example, if I want to flip all my images, my dataset is equivalent to multiplying by 2.

**Before:**

<img src="./img/unbalanced_data.png" alt="Description of your image" width=“200” height="300">

<!-- ![Alt text](./img/unbalanced_data.png) -->

**After:**

<img src="./img/balanced.png" alt="Description of your image" width=“200” height="200">


### Methodology
We did reszie, flip, random crop, rotation and colorJitter to the image to augment and get a larger dataset.
1.  `train_transform_1`:

-   `transforms.Resize((image_height, image_width))`: This transformation resizes the input image to the specified height and width. It's often used to standardize the size of input images for a neural network.

2.  `train_transform_2`:

-   `transforms.RandomHorizontalFlip()`: This randomly flips the image horizontally with a default 50% probability. It's useful for augmenting image datasets where the orientation isn't crucial.

3.  `train_transform_3`:

-   `transforms.RandomRotation(10)`: This randomly rotates the image by a degree selected from a uniform distribution within the range [-10, 10] degrees. It adds variability in terms of rotation, making the model more robust to different orientations.

4.  `train_transform_4`:

-   `transforms.RandomResizedCrop((image_height, image_width), scale=(0.8, 1.0), ratio=(0.9, 1.1))`: This applies a random resized crop to the image. It first randomly scales the image and then crops it to the specified size. The `scale` and `ratio` parameters control the variability in size and aspect ratio, respectively, of the crop.

5.  `train_transform_5`:

-   `transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)`: This randomly changes the brightness, contrast, saturation, and hue of the image. The parameters control the degree of jittering for each attribute.


### Model Train

1. Machine Learning model - N-gram Model

The n-gram model is a type of language model that predicts the next word by statistically analyzing the frequency of sequences of n words (i.e., n-grams) in the text. For example, a 2-gram (or bigram) model would consider sequences of every two words, while a 3-gram (or trigram) model would consider sequences of every three words. The steps is summarized below:

1. **Data Loading and Processing**: Through the `load_data` method, poetry data is loaded from a specified CSV file, containing the titles and lyrics of the songs. The loaded data is stored in a Pandas DataFrame for easy subsequent processing.

2. **Text Tokenization**: In the `tokensize` method, all lyrics are first combined into a long string, then tokenized into a list of words using the `nltk` `word_tokenize` method. This step converts the text into a sequence of words that can be further analyzed.

3. **Removing Stop Words and Non-alphabetic Characters**: The project further processes the tokenized words, removing English stop words and non-alphabetic characters to reduce noise and focus on content words. This is achieved by checking if each token is alphabetic and not in the list of stop words.

4. **Generating Bigrams**: Using the `generate_bigrams` method, the cleaned word sequence is converted into bigrams (word pairs), providing the model with the ability to consider the adjacent relationships between words. At the same time, the frequency distribution of bigrams is calculated and stored, which is very useful for understanding the relationships between words and generating text.

5. **Text Generation**: The `generate_text` method starts by randomly selecting a bigram as the starting point. Then, based on the second word of the current bigram, it finds all possible subsequent words and randomly selects one as the next word, and so on, generating a sequence of words. This method can generate text sequences that appear relatively coherent in grammar and semantics, although the generated text may have some randomness and uncertainty.


**N-gram Model result**

>see way well pressing yes believe man comes man peace information name give back town cold frosty morn creeps avoidin southside best soon lost race goes babylon girl france invited house fire peered darkness away kicked neighborhood bully fall well already gone spirit maker heartbreaker backbreaker leave know sun strong monkey log



2. Deep Learning model - VIT (Vision Transformer)

The Vision Transformer (ViT) model adopts an innovative approach to apply the Transformer architecture to image classification tasks. In ViT, the input image is first divided into fixed-size small patches, similar to dividing text into words or subwords in natural language processing. To retain positional information, ViT introduces positional embeddings, which are added to the representations of the image patches before being fed into a standard Transformer model.

The core of the Transformer model is the self-attention mechanism, which allows the model to consider information from all patches in the image while processing each patch, thereby capturing long-distance dependencies. The ability of this structure to directly utilize global information is one of the main advantages of ViT over traditional convolutional networks.

![Alt text](./img/vit_architecture.jpg)

**Fine-tuned VIT Model result**
1. F1 score: 0.985
2. Confusion Matrix:
![Alt text](./img/ViT_cf.png)

### Model Evaluation

We chose ChatGPT-4 as our third-party model to judge which output is closer to Bob Dylan's poetry style. We called the GPT-4 API and set an appropriate instruct prompt, using the generated poems from the two models to be compared as inputs. Then, we tallied the assessment results from GPT-4.

#### Model Comparison
| Model          | F1 score | Running Time |
|---------------|----------------------------------| -------------------|
| SVM     | 0.652   | 1:17:45 (20 epoch) |
| VIT    | 0.985  | 4:01 |

### Inference
We built a web interface using `streamlit`. You can input an image of an artwork, and it will attempt to predict the year in which the artwork was created.

![Alt text](./img/interface.png)


The result of the prediction
![Alt text](./img/show.png)





