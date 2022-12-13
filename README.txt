1. Models đã train ở link (5 model tương ứng 5 fold)
	https://drive.google.com/drive/folders/15wVBcUameA-n8Pj9iNJePQppdCVubsdh?fbclid=IwAR1xAPFFp5yOXqbROQd6sLaGmAXoS7XHO1IYUBqoRzBoFs7UKI_NCZOHlQw
2. Clone git, cd đến folder vừa clone
3. Run requirements.txt
4. Run: 

!mkdir -p vncorenlp/models/wordsegmenter
!wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
!wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
!wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
!mv VnCoreNLP-1.1.1.jar vncorenlp/ 
!mv vi-vocab vncorenlp/models/wordsegmenter/
!mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/

!wget https://public.vinai.io/PhoBERT_base_transformers.tar.gz
!tar -xzvf PhoBERT_base_transformers.tar.gz

5. Sửa link dict.txt ở file infer.py (dòng 55) và file train.py (dòng 68)

6. Paste và sửa link phù hợp 
- Train:

!python train.py --fold 0 \
--train_path '/content/drive/MyDrive/Colab_Notebooks/Phobert/PhoBert-Sentiment-Classification/full_train.csv' \
--dict_path "./PhoBERT_base_transformers/model.bin" \
--config_path "./PhoBERT_base_transformers/config.json" \
--bpe-codes "./PhoBERT_base_transformers/bpe.codes" \
--pretrained_path './PhoBERT_base_transformers/model.bin' \
--ckpt_path './models' \
--rdrsegmenter_path "/content/drive/MyDrive/Colab_Notebooks/Phobert/PhoBert-Sentiment-Classification/vncorenlp/VnCoreNLP-1.1.1.jar"

**** Train 5 lần, sửa --fold lần lượt từ 0 đến 4, 
     Mỗi lần run train, khi tiến độ hiển thị     100% 6/6 [1:16:20<38:10, 1145.05s/it]    thì ấn stop run và thay số --fold train tiếp  

- Test:

!python infer.py --fold x \
--test_path '/content/drive/MyDrive/Colab Notebooks/Phobert/Sentiment_Phobert/test.csv' \
--dict_path "./PhoBERT_base_transformers/model.bin" \
--config_path "./PhoBERT_base_transformers/config.json" \
--bpe-codes "./PhoBERT_base_transformers/bpe.codes" \
--pretrained_path './PhoBERT_base_transformers/model.bin' \
--ckpt_path './models' \
--rdrsegmenter_path "/content/drive/MyDrive/Colab Notebooks/Phobert/Sentiment_Phobert/vncorenlp/VnCoreNLP-1.1.1.jar"

**** Thay x = số fold đã train (các số từ 1 đến 5, khuyến khích train đủ 5 lần)
     Khi thấy hiển thị: 
	Predicting for fold 0
	Predicting for fold 1
	Predicting for fold 2
	Predicting for fold 3
	Predicting for fold 4
     thì stop run và download file submission.csv

**** Nếu muốn dùng model train sẵn, tạo folder Models, tải và paste Model ở bước 1 vào folder Models.

7. Nếu muốn chạy code clean data: 
  + Xoá # ở file infer.py (dòng 74) và file train.py (dòng 76, 77)
  + Sửa path_nag, path_pos (dòng 118, 119)
