import os
import quilt3

# download from AWS bucket
b = quilt3.Bucket("s3://asem-project")

b.fetch("datasets/cell_1/cell_1.zarr/", "cell_6/cell_6.zarr/")
# load database dump into local mongodb
# os.popen('mongorestore --archive="models/fiborganelle_trainings" --nsFrom="public_fiborganelle.*" --nsTo="incasem_trainings.*"').read()
