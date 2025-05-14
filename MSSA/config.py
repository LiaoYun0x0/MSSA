import yaml
import argparse

# 加载配置文件
def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='G:\\Image-Text_Matching\\work1all\\rs\\option\\RSITMD.ymal', type=str,
                      help='path to a yaml options file') 

    opt = parser.parse_args()

    # load model options
    with open(opt.path_opt, encoding='utf-8') as f:
        options = yaml.safe_load(f)
    return options



    # Host i-1.gpushare.com
#   HostName i-1.gpushare.com
#   Port 1260
#   User root


# Host i-1.gpushare.com
#   HostName i-1.gpushare.com
#   Port 51552
#   User root