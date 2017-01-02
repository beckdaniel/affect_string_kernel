
TRIAL_SENTS=../../data/AffectiveText.trial/affectivetext_trial.xml
TRIAL_EMOS=../../data/AffectiveText.trial/affectivetext_trial.emotions.gold
TRIAL_VAL=../../data/AffectiveText.trial/affectivetext_trial.valence.gold
TEST_SENTS=../../data/AffectiveText.test/affectivetext_test.xml
TEST_EMOS=../../data/AffectiveText.test/affectivetext_test.emotions.gold
TEST_VAL=../../data/AffectiveText.test/affectivetext_test.valence.gold

cat $TRIAL_EMOS $TEST_EMOS > ../../data/emotions.gold
cat $TRIAL_VAL $TEST_VAL > ../../data/valence.gold

head -251 $TRIAL_SENTS | tail -250 | sed 's/<instance id="//g' | sed "s|</instance>||g" | sed 's/">/_/g'> ../../data/AffectiveText.trial/instances.txt
head -1001 $TEST_SENTS | tail -1000 | sed 's/<instance id="//g' | sed "s|</instance>||g" | sed 's/">/_/g' > ../../data/AffectiveText.test/instances.txt
cat ../../data/AffectiveText.trial/instances.txt ../../data/AffectiveText.test/instances.txt > ../../data/instances.txt

