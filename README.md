# kubix-square

Topic: Sentiment analysis on Cryptocurrency news which will be updated on Twitter. (top 10 cryptocurrencies will be taken into consideration according to their market cap).

- 10 Machine Learning models were trained on the .csv file uploaded on the repo.
- Out of those, the ones which had an accuracy greater than 60% were downloaded using the Pickle library. 
- In the final code for the frontend website, we have used the Coinbase API to get the list of the top 10 cryptocurrencies based on their market cap. 
- This list is then used to extract raw, live tweets from Twitter using the Twitter API.
- These tweets are preprocessed and fed into the selected 4 Machine Learning models. 
- Each model outputs a sentiment value (-1, 0, 1) for each tweet, and the output from each model is compared and the majority is the final output. e.g if the output from all the 4 models is (0,1,1,1) then the final output is 1 i.e. Positive.
- The final output is displayed in a bar graph manner which the x-axis being the sentiments and the y-axis being the number of tweets of each senitment.
- internship.ipynb is the Jupyter Notebook which contains the code for cleaning, preprocessing the data, and training and testing of the various Machine Learning models.
- mk1.py is the backend code for the website, and requirement.txt is a notepad file listing all the Python libraries used along with their version number. 
 
Jash Tailor <br>
LinkedIn: https://www.linkedin.com/in/jashtailor/ <br>
Github: https://github.com/jashtailor

Beryl Coutinho <br>
LinkedIn: https://www.linkedin.com/in/beryl-coutinho/ <br>
Github: https://github.com/BerylCoutinho

Cyrus Ferreira <br>
LinkedIn: https://www.linkedin.com/in/cyrus-ferreira-209763181/ <br>
Github: https://github.com/cyrusf17

Shubh Joshi <br>
LinkedIn: https://www.linkedin.com/in/shubh-joshi-481abb1ba <br>
Github: https://github.com/ShubhJoshi-557

Arnold Veigas <br>
LinkedIn: https://www.linkedin.com/in/arnold-veigas-390460217 <br>
Github: https://github.com/Arnold17V

Frontend: https://share.streamlit.io/jashtailor/kubix-square/main/mk1.py
