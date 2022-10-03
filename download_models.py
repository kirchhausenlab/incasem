import os
import quilt3

# download from AWS bucket
b = quilt3.Bucket("s3://asem-project")
b.fetch("models/", "./models/")

# load database dump into local mongodb
os.popen('mongorestore --archive="models/fiborganelle_trainings" --nsFrom="public_fiborganelle.*" --nsTo="incasem_trainings.*"').read()
