import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from lib.test.analysis.plot_results import print_results,print_results_per_video
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'lasot'
### him2sam
trackers.extend(trackerlist(name='him2sam', parameter_name='him2sam_large', dataset_name=dataset_name,
                            run_ids=None, display_name='him2sam_large'))
trackers.extend(trackerlist(name='him2sam', parameter_name='him2sam_base_plus', dataset_name=dataset_name,
                            run_ids=None, display_name='him2sam_base_plus'))
trackers.extend(trackerlist(name='him2sam', parameter_name='him2sam_small', dataset_name=dataset_name,
                            run_ids=None, display_name='him2sam_small'))
trackers.extend(trackerlist(name='him2sam', parameter_name='him2sam_tiny', dataset_name=dataset_name,
                            run_ids=None, display_name='him2sam_tiny'))
                              
                          

# trackers.extend(trackerlist(name='samurai', parameter_name='samurai_large', dataset_name=dataset_name,
#                             run_ids=None, display_name='samurai_large'))
# trackers.extend(trackerlist(name='samurai', parameter_name='samurai_base_plus', dataset_name=dataset_name,
#                             run_ids=None, display_name='samurai_base_plus'))
# trackers.extend(trackerlist(name='samurai', parameter_name='samurai_small', dataset_name=dataset_name,
#                             run_ids=None, display_name='samurai_small'))
# trackers.extend(trackerlist(name='samurai', parameter_name='samurai_tiny', dataset_name=dataset_name,
#                             run_ids=None, display_name='samurai_tiny'))


dataset = get_dataset(dataset_name)

print_results(trackers, dataset, dataset_name, merge_results=False, plot_types=('success','AUC', 'norm_prec', 'prec'))

print_results_per_video(trackers, dataset, dataset_name, merge_results=False, plot_types=('success','AUC', 'norm_prec', 'prec'),per_video=True)