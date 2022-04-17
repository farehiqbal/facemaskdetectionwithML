# Face Mask Detection using Python and Machine Learning.

- Created a fully fuctionaling face mask detector.

- HAAR CASCADE algorithm is used in the CV2 module which determines if there is a face in frame or not.


# NOTE:
This is not the final version, Its still a draft to understand the working, I might update it in the future to add convenience to the program and get rid of uncommenting to enter the data.

- In my case, the algorithm was 97% effective, it might differ in your case. There is a module which displays the efficieny of the algorithm.


## Requriements:

- Python.
- Understadning of CV2 module.
- Basic knowledge of numpy.

## Working:

It works on the principle of machine learning. It collects data (with mask & withoutmask) using your webcam, and then compares this data to the real-time webcam to show
if there is face on the face or not.

## How To Use:

- Install all the requirements.
- Uncomment the [with_mask.npy] and provide frame with mask to the program.
- Comment the [with_mask.npy] and uncomment [without_mask.npy] and provide frames or you can do vice versa.
- Run the file 2 and it will now show you if you have mask on or not.


