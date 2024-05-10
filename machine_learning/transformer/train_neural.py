import sys, traceback
from machine_learning.transformer.training import model_training

def train_neural(content_space_ids):
    for content_space_id in content_space_ids:
        try:
            model_training(content_space_id=content_space_id, training=True)
        except:
            print(traceback.format_exc(), file=sys.stderr)
            # TODO: Generate 'report' of exceptions for all content spaces?
            continue
