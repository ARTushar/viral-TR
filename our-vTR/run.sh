case $1 in
    h)
        echo 'running hypertuner'
        python3 hypertuner.py
    ;;
    c)
        echo 'running CV'
        python3 CV.py
    ;;
    t)
        echo 'running plain train'
        python3 train.py
    ;;
    b)
        case $2 in
        '')
            echo 'enter fold count'
            exit
        ;;
        *)
            echo 'running tensorboard'
            python3 -m tensorboard.main --logdir $2_fold_lightning_logs/dataset1
        ;;
        esac
    ;;
    a)
        echo 'running aggregator'
        python3 utils/tb_aggregator.py -d 'lightning_logs/*' -o 'lightning_logs/test_aggr'
    ;;
    *)
        echo 'invalid arguments, check inside' $0 'file'
    ;;
esac
