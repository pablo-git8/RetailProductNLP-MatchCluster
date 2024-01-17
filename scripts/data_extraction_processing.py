# Imports
import pandas as pd
import pandas as pd
from text_tokenization import tokenizeText


### DATA EXTRACTION ###

# Retailer A data
retA_data = pd.read_csv('../data/raw/retailerA.csv')
# Retailer B data
retB_data = pd.read_csv('../data/raw/retailerB.csv')
# Combining data from RetA and RetB
combined_data = pd.concat([retA_data, retB_data])


### DATA PROCESSING ###

# Create column in dataset with tokenized 'title'
retA_data['title_token'] = retA_data['title'].apply(tokenizeText, args=(False,))
retB_data['title_token'] = retB_data['title'].apply(tokenizeText, args=(False,))
combined_data['title_token'] = combined_data['title'].apply(tokenizeText, args=(False,))

# Saving intermedite datasets
retA_data[['title', 'title_token']].to_csv('../data/raw/retailerA_tokens.csv')
retB_data[['title', 'title_token']].to_csv('../data/raw/retailerB_tokens.csv')
combined_data[['title', 'title_token']].to_csv('../data/raw/combined_data_tokens.csv')