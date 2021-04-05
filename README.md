# Assignment 4: MAIS202: Kaggle Competition
## MAIS202 Winter2021 Assignment 4: Find the maximum digit in an image
### We implemented the VGG16 architecture for our model. Here is how you can reproduce our results

[Run the jupyter notebook from this repo](./MAIS202_A4_Competition/src/training.ipynb) 
or [Run the trained model by downloading it from this link](https://drive.google.com/file/d/1Ns4Y8ibEQFHZkEBtZrLvbSMzWNWG3TRx/view?usp=sharing)

### Steps to run the Notebook from this Repo
1) Initial steps

> Running the first cell will download all the data required.

> Run cell 2 to import all the required libraries.

> Run cell 3 to load the data as numpy arrays.

> Run cell 4 to mount to your google drive.

> Run cell 5 to print some images from the training set.

> Finally run cell 6 to load the training images and cells under “Preparing the data” to prep the dataset.

2) Implementing the model

> Run the cell under “Defining the model” to define our implementation of the VGG16 architecture.

3) Training

> Run all the cells under “Training” to train the model and save it to your drive.

4) Testing

> Run the cells under “Making inferences on test set” to test the model. 

> This will save the test results to a csv file on your drive. Download it to see the results!


### Steps to run the trained model from the provided link

1) Download it 
2) Load it in you notebook using 

```
model = pt.load(path_to_the_downloaded_model)
```

3) Load the testing images using
```
!wget -O test_x.npy  https://www.dropbox.com/s/qfbaw6a18cthkg4/test_x.npy?dl=0
test_images = np.load("test_x.npy")

```
4) Make inferences using:
```
from tqdm import tqdm 

def inference(model, test_data, file_path):
    model.eval()
    device = pt.device('cpu')
    if pt.cuda.is_available():
        device = pt.device('cuda')
    model.to(device)
    inferences = [["ID", "Category"]]
    for i, img in enumerate(tqdm(test_data)):
        input = pt.tensor(img).unsqueeze(0).unsqueeze(0)
        input = input.to(device)
        out = int(pt.argmax(model.forward(input)))
        inferences.append([i, out])

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(inferences)

inference(model, test_images_processed, "/content/MNIST_inferences.csv")
```
5) Downlad the csv files with the results!

