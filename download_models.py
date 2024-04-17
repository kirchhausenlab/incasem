import os
import quilt3

# download from AWS bucket
b = quilt3.Bucket("s3://asem-project")
b.fetch("models/", "/workspace/src/incasem/models/")
