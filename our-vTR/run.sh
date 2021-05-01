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
    *)
        echo 'invalid arguments, check inside' $0 'file'
    ;;
esac
