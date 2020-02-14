# Glassdoor Review (an ongoing project)

### Scripts and notebooks
* script: read_large_dta.py: reads the original 16GB STATA data file, and randomly select a representative subsample to conduct text analysis
* script: data_preprocessing.py: pre-process reviews on company pros and cons by: removing stop words + lemmatization
* script: LDA_ntopics.py: trying LDA model with different number of topics and plot the coherence scores to find optimal number of topics to set for LDA model; trained LDA models with optimal number of topics
* notebook: LDA_visualization.ipynb: visualized the topics found by LDA using pyLDAvis
* script: label_topics.py: label topics to reviews using trained LDA model

### Topic visualization
The screenshot below exhibits a visualization of a topic found by the LDA model of all the employee reviews about cons.<br><br>
![png](graphs/pyLDAvis_example1.png)
<br><br>
### Topic labeling
The topics are then hand labeled according to the associated word frequency. <br> 
#### Topics of reviews on pros: 
* Salary and Benefits
* Flexible Schedule
* Career Opportunity
* Work-Life Balance
* Supportive Management
* Culture and Value
* Food and Perks
* Friendly and Smart Colleagues
* Friendly to Juniors
#### Topics for reviews on cons:
* Low Pay and High Turnover Rate
* Long Working Hours
* Limited Career Opportunity
* Demanding Work
* Bad Manager
* Poor Communication
* Pressure from Sales and Customer Service
* Slow to Adapt to Changes
### Compare topic distribution across various companies (in progress)
Screenshots from a work-in-progress Tableau dashboard:<br><br>
![png](graphs/topic_pros.png) <br><br>
![png](graphs/topic_cons.png) <br><br>


