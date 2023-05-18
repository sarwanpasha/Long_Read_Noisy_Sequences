from Bio import SeqIO
from Bio.Seq import Seq
import os

i=1
for seq_record in SeqIO.parse("/alina-data2/Bikram/GISAID/pbsim_pipeline/gisaid_hcov-19.fasta","fasta"):
	my_seq = seq_record.seq
	my_id = seq_record.id
	
	seq = ">"+my_id+"\n"+my_seq
	seq_ = "seq_"+str(i)+".fasta"
	f = open(seq_,"w")
	f.write(str(seq))

	cmd = 'badread simulate --reference seq_'+str(i)+'.fasta --quantity 10x --error_model random --qscore_model random --identity 85,95,3 --length 7500,7500 | gzip > sd_0001.fastq.gz'
	os.system(cmd)

	cmda = 'gzip -d sd_0001.fastq.gz'
	os.system(cmda)

	cmd1 = 'mv sd_0001.fastq sd_'+str(i)+'.fastq'
	os.system(cmd1)

	cmd2 = 'rm seq_'+str(i)+'.fasta'
	os.system(cmd2)

	cmd3 = '/alina-data2/Bikram/GISAID/pbsim_pipeline/minimap2-2.24_x64-linux/minimap2 -a /alina-data2/Bikram/GISAID/NC_0455122.fa sd_'+str(i)+'.fastq'+'>'+'aln_'+str(i)+'.sam'
	os.system(cmd3)

	cmd4 = 'rm sd_'+str(i)+'.fastq' 
	os.system(cmd4)

	cmd5 = 'samtools view -bS aln_'+str(i)+'.sam > aln_'+str(i)+'.bam'
	os.system(cmd5)

	cmd6 = 'rm aln_'+str(i)+'.sam'
	os.system(cmd6)

	#cmd_6 = 'samtools sort -n -@ 4 aln_'+str(i)+'.bam -o aln_'+str(i)+'.sorted.bam'
	#os.system(cmd_6)

	cmd_6 = 'java -jar /alina-data2/Bikram/New_data_IS/COVID19/picard.jar SortSam I=aln_'+str(i)+'.bam O=aln_'+str(i)+'.sorted.bam SORT_ORDER=coordinate'
	os.system(cmd_6)

	cmd7 = 'bcftools mpileup -Ou -f /alina-data2/Bikram/GISAID/NC_0455122.fa aln_'+str(i)+'.sorted.bam | bcftools call -Ou -mv | bcftools norm -f /alina-data2/Bikram/GISAID/NC_0455122.fa -Oz -o output_'+str(i)+'.vcf.gz'
	os.system(cmd7)

	cmd_7 = 'rm aln_'+str(i)+'.sorted.bam'
	os.system(cmd_7)

	cmd8 = 'rm aln_'+str(i)+'.bam'
	os.system(cmd8)

	cmd9 = 'tabix output_'+str(i)+'.vcf.gz'
	os.system(cmd9)

	cmd10 = 'bcftools consensus -f /alina-data2/Bikram/GISAID/NC_0455122.fa output_'+str(i)+'.vcf.gz > out_'+str(i)+'.fa'
	os.system(cmd10)

	cmd11 = 'rm output_'+str(i)+'.vcf.gz'
	os.system(cmd11)

	cmd12 = 'cat out_'+str(i)+'.fa >> random_10x_simulated_error.fasta'
	os.system(cmd12)

	cmd13 = 'rm out_'+str(i)+'.fa'
	os.system(cmd13)

	cmd14 = 'rm output_'+str(i)+'.vcf.gz.tbi'
	os.system(cmd14)

		
			
	i = i+1
