
## Spam Detection Using Logistic Regression


# This program classifies emails into two categories: span and no-spam. It uses logistic regression for 
# classification. The training data are two files which are already classified.
# This program will work for any new text file containing emails and will classify them as 
# spam or no-spam categories.
#############################################################################################################


#############################################################################################################
# Importing nessesary libraries and functions 
#############################################################################################################

from pyspark.sql import SparkSession
import sys
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.feature import HashingTF

reload(sys)
sys.setdefaultencoding('utf8')

if __name__ == "__main__":
	
	print ("This is the name of the script: ", sys.argv[0])
	print ("Number of arguments: ", len(sys.argv))
	print ("The arguments are: " , str(sys.argv))
	
	if len(sys.argv) != 5:
		print("Usage: filenamewithlocation path nospamfile spamfile queryfile")
		exit(-1)
	
	#############################################################################################################
	# Get all the information from system arguments such as file path, all files names used in program
	# Argument 1: Path where all files are stored
	# Argument 2: no spam file name (training data)
	# Argument 3: spam file name (training data)
	# Argument 4: query file name (test data)
	#############################################################################################################
	file_path = sys.argv[1]
	nospam = sys.argv[2]
	spam = sys.argv[3]
	query = sys.argv[4]
	
	#############################################################################################################
	# Create spark session object
	#############################################################################################################

	spark = SparkSession\
			.builder\
			.appName("SpamDetection")\
			.getOrCreate()

	#############################################################################################################
	# Read the two training data files
	# 
	# Training data  consist of two sets , spam and nospam text files. Both Contains 20 records of spam and 20 records of no spam. 
	# We first create spark context and then we read the files.

	# Create spark context
	#############################################################################################################

	sc = spark.sparkContext

	nospam=sc.textFile(file_path + '\\' + nospam)
	spam=sc.textFile(file_path + '\\' + spam)


	#############################################################################################################
	# Next to build logistic model from the given training data , we follow the following steps :
	# Create a feature vector with 1000 features using HashingTF class
	#############################################################################################################

	featurize_data = HashingTF(numFeatures= 1000)

	#############################################################################################################
	# Next, we need to map these features with our datasets. First we split each email into words
	# Then, we will use HashingTF  to find the frequency of word in the mail and create a Vector.
	#############################################################################################################

	featurize_spam =spam.map(lambda x:featurize_data.transform(x.split(" ")))
	featurize_nospam=nospam.map(lambda x:featurize_data.transform(x.split(" ")))

	############################################################################################################# 
	# Create labeled Point:
	# To provide labeded data to the application, we will create labeled points.A labeled point is a local vector, 
	# either dense or sparse, associated with a label/response.
	# 
	# We are going to label spam as 1's and nospam as 0's.  We create label points as follows:
	#############################################################################################################

	positive_spam= featurize_spam.map(lambda x:LabeledPoint(1,x))
	negative_spam= featurize_nospam.map(lambda x:LabeledPoint(0, x))

	#############################################################################################################
	# Now we club both the Label points and also cache RDD as we use it repeatedly
	#############################################################################################################

	training_set= positive_spam.union(negative_spam)
	training_set.cache()

	#############################################################################################################
	# Finally, we run Logistic Regression  using Stochastic Gradient Descent(SGD) on training data.
	# 
	# Stochastic Gradient Descent**(SGD) : SGD works by using the model to calculate a prediction 
	# for each instance in the training set and calculating the error for each prediction. 
	# The process is repeated until the model is accurate enough (e.g. error drops to some desirable level) 
	# or for a fixed number iterations. You continue to update the model for training instances and correcting 
	# errors until the model is accurate enough or cannot be made any more accurate.
	#############################################################################################################

	model=LogisticRegressionWithSGD.train(training_set)

	#############################################################################################################
	# Now we use the built logistic model to read new set of emails called query data (test data) and the classify 
	# them as "spam" (label 1) or "non-spam" (label 0). We use the following steps to implement the same:
	# 
	# Read the query data file:
	#############################################################################################################

	query=sc.textFile(file_path + '\\' + query)

	#############################################################################################################
	# Featurize query rdd; First we split each email is  into words; Then, we will use HashingTF  to 
	# find the frequency of word in the mail and create a Vector
	#############################################################################################################


	featurize_query=query.map(lambda x:featurize_data.transform(x.split(" ")))

	#############################################################################################################
	# Classify the featurize_query and then using map emit classification and the corresponding mail.
	#############################################################################################################


	pred_query = model.predict(featurize_query)
	classifiaction_email = pred_query.zip(query.map(lambda x: x))

	#############################################################################################################
	# Finally, we measure the accuracy percentage of the model by using spams.txt and nonspams.txt as a query.
	# (Here we use the same featurized training_set as input to the model)
	#############################################################################################################

	##Classify the training set using model:
	pred = model.predict(training_set.map(lambda x: x.features))

	##zip the predicted labels and actual labels:
	actual_pred = training_set.map(lambda x: x.label).zip(pred)

	# overall accuracy of the model 
	Overall_accuracy = actual_pred .filter(lambda x: x[0] == x[1]).count() / float(training_set.count())

	#Spam Accuracy of Model
	Spam_accuracy = actual_pred.filter(lambda x: x[0] == 1).count() / float(training_set.filter(lambda x: x.label==1).count())
	Nospam_accuracy = actual_pred.filter(lambda x: x[0] == 0).count() / float(training_set.filter(lambda x: x.label==0).count())

	out_text =  classifiaction_email.collect()
	
	###############################################################################################################
	# Output 1: <classification> <email>
	###############################################################################################################
	print "#####################################################################################################################"
	print "########################################## START OUTPUT #############################################################"
	
	for classification, email_text in out_text:
		email_text = email_text.decode('ascii', 'ignore')
		print '%s -- %s' % (classification, email_text)

	###############################################################################################################
	# Output 2:  Overall Accuracy of Model: <%>
    # Spam Accuracy of Model: <%>
    # Non-Spam Accuracy of Model: <%>
	###############################################################################################################
	
	print "Overall accuracy of the model is:%g%s"%(Overall_accuracy*100,'%')
	print "Spam Accuracy of Model:%g%s"%(Spam_accuracy*100,'%')
	print "Non-Spam Accuracy of Model:%g%s"%(Nospam_accuracy*100,'%')
	
	print "########################################## END OUTPUT #################################################################"
	print "#######################################################################################################################"
	
	###############################################################################################################
	# Stop the spark context
	###############################################################################################################
	sc.stop()

