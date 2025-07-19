@gabigaudeau, I have created a Google Drive Folder with the raw training data that needs to be preprocessed: https://drive.google.com/drive/folders/16G27TS_91hIKxJWFonrb5yO8X9FFBJ0k?usp=sharing 

See the Overleaf for word counts, which I have estimated should be equal to ~70M token.

We need to combine each of these files into a single txt file, and then tokenise and shuffle to pretrain the OPT BabyLMs. 

## Some Potentially Useful Scripts

See preprocess.py for some potentially useful preprocessing things.
