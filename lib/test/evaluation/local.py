from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.lasot_path = './data/LaSOT'
    settings.lasot_extension_subset_path = './data/LaSOT_extension_subset'
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.uav_path = ''
    settings.results_path = './results'
    settings.result_plot_path = './evaluation_results'
    settings.save_dir = './evaluation_results'

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/baiyifan/code/OSTrack/data/got10k_lmdb'
    settings.got10k_path = '/home/baiyifan/GOT-10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/baiyifan/code/OSTrack/data/itb'
    settings.lasot_lmdb_path = '/home/baiyifan/code/OSTrack/data/lasot_lmdb'
    settings.network_path =  '/ssddata/baiyifan/artrack_256_full_re/'   # Where tracking networks are stored.
    settings.prj_dir = '/home/baiyifan/code/2d_autoregressive/bins_mask'
    settings.segmentation_path = '/data1/os/test/segmentation_results'
    settings.tc128_path = '/home/baiyifan/code/OSTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/baiyifan/code/OSTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/ssddata/TrackingNet/all_zip'
    settings.vot18_path = '/home/baiyifan/code/OSTrack/data/vot2018'
    settings.vot22_path = '/home/baiyifan/code/OSTrack/data/vot2022'
    settings.vot_path = '/home/baiyifan/code/OSTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

