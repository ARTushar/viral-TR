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
        'n')
            echo 'running tensorboard'
            python3 -m tensorboard.main --logdir ../globals/lightning_logs
        ;;
        *)
            echo 'running tensorboard'
            python3 -m tensorboard.main --logdir ../globals/$2_fold_lightning_logs/normal/normal/SRR5241430
        ;;
        esac
    ;;
    a)
        echo 'running aggregator'
        python3 utils/tb_aggregator.py -d '../globals/lightning_logs/default/*' -o '../globals/lightning_logs/test_aggr' -f
    ;;
    r)
        case $3 in
        '')
            echo 'enter fold count & version number'
            exit
        ;;
        *)
            for folder in $(find $2_fold_lightning_logs -type d -name 'version_'$3); do
                echo $folder
                # rm -rf $folder
            done
        ;;
        esac
    ;;
    *)
        echo 'invalid arguments, check inside' $0 'file'
    ;;
esac
