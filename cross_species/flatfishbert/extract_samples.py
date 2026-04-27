#!/usr/bin/env python3
"""
泛化物种数据提取脚本
从3个新物种的GFF+FASTA中提取功能区序列，格式与V3一致
"""

import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from Bio import SeqIO
from Bio.Seq import Seq
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import SPECIES_CONFIGS, DATA_OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeneralizationDataExtractor:
    """泛化物种数据提取器"""

    TRANSCRIPT_TYPES = {'mRNA', 'lnc_RNA', 'transcript', 'ncRNA', 'primary_transcript'}
    DIRECT_FEATURE_TYPES = {'CDS', 'tRNA', 'rRNA', 'snRNA', 'snoRNA'}

    def __init__(self):
        self.unknown_gene_counter = 0

    def parse_gff_attributes(self, attr_str):
        """解析GFF属性字段"""
        attrs = {}
        for item in attr_str.split(';'):
            item = item.strip()
            if '=' in item:
                key, value = item.split('=', 1)
                attrs[key.strip()] = value.strip()
        return attrs

    def generate_unknown_gene_name(self, attrs, seqname, start, end):
        """为未知基因生成唯一标识符"""
        feature_id = attrs.get('ID', '')
        if feature_id:
            if feature_id.startswith('cds-'):
                return f"unknown_cds_{feature_id[4:]}"
            elif feature_id.startswith('exon-'):
                return f"unknown_exon_{feature_id[5:]}"
            elif feature_id.startswith('rna-'):
                return f"unknown_rna_{feature_id[4:]}"
            elif feature_id.startswith('gene-'):
                return feature_id[5:]
            else:
                return f"unknown_{feature_id}"
        self.unknown_gene_counter += 1
        return f"unknown_{seqname}_{start}_{end}_{self.unknown_gene_counter}"

    def get_gene_name(self, attrs, seqname=None, start=None, end=None):
        """从属性中提取基因名"""
        gene_name = attrs.get('gene', '').strip()
        if gene_name:
            return gene_name
        name = attrs.get('Name', '').strip()
        if name:
            return name
        feature_id = attrs.get('ID', '')
        if feature_id and feature_id.startswith('gene-'):
            extracted = feature_id[5:].strip()
            if extracted:
                return extracted
        parent = attrs.get('Parent', '')
        if parent and parent.startswith('gene-'):
            extracted = parent[5:].strip()
            if extracted:
                return extracted
        return self.generate_unknown_gene_name(attrs, seqname, start, end)

    def parse_gff(self, gff_path, species_name):
        """解析GFF文件，提取所有功能区"""
        logger.info(f"解析GFF: {gff_path}")

        features = {ft: [] for ft in self.DIRECT_FEATURE_TYPES}
        features['lncRNA_exon'] = []
        gene_info = {}
        transcript_exons = defaultdict(list)
        transcript_info = {}
        lncrna_exons = defaultdict(list)
        lncrna_info = {}

        line_count = 0
        with open(gff_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue

                line_count += 1
                seqname = parts[0]
                feature_type = parts[2]
                start = int(parts[3])
                end = int(parts[4])
                strand = parts[6]
                attrs = self.parse_gff_attributes(parts[8])

                if feature_type == 'gene':
                    gene_id = attrs.get('ID', '')
                    gene_name = attrs.get('gene', attrs.get('Name', ''))
                    if not gene_name and gene_id.startswith('gene-'):
                        gene_name = gene_id[5:]
                    gene_info[gene_id] = gene_name if gene_name else 'unknown'

                elif feature_type in self.TRANSCRIPT_TYPES:
                    transcript_id = attrs.get('ID', '')
                    parent = attrs.get('Parent', '')
                    gene_name = gene_info.get(parent, '')
                    if not gene_name:
                        gene_name = self.get_gene_name(attrs, seqname, start, end)
                    transcript_info[transcript_id] = {
                        'gene_name': gene_name,
                        'seqname': seqname,
                        'strand': strand,
                        'type': feature_type
                    }

                elif feature_type in self.DIRECT_FEATURE_TYPES:
                    gene_name = self.get_gene_name(attrs, seqname, start, end)
                    features[feature_type].append({
                        'seqname': seqname, 'start': start, 'end': end,
                        'strand': strand, 'gene_name': gene_name,
                        'feature_type': feature_type
                    })

                elif feature_type == 'exon':
                    parent = attrs.get('Parent', '')
                    gene_name = self.get_gene_name(attrs, seqname, start, end)

                    if parent.startswith('rna-XR_'):
                        lncrna_exons[parent].append((start, end))
                        if parent not in lncrna_info:
                            if parent in transcript_info:
                                gene_name = transcript_info[parent]['gene_name']
                            lncrna_info[parent] = {
                                'gene_name': gene_name,
                                'seqname': seqname,
                                'strand': strand
                            }

                    if parent in transcript_info:
                        transcript_exons[parent].append((start, end))

        logger.info(f"  解析了 {line_count:,} 行GFF记录")

        # 添加lncRNA exon
        for lncrna_id, exons in lncrna_exons.items():
            info = lncrna_info[lncrna_id]
            for start, end in exons:
                features['lncRNA_exon'].append({
                    'seqname': info['seqname'], 'start': start, 'end': end,
                    'strand': info['strand'], 'gene_name': info['gene_name'],
                    'feature_type': 'lncRNA_exon', 'transcript_id': lncrna_id
                })

        # 计算intron
        introns = []
        for transcript_id, exons in transcript_exons.items():
            if len(exons) < 2:
                continue
            info = transcript_info[transcript_id]
            exons_sorted = sorted(exons, key=lambda x: x[0])
            for i in range(len(exons_sorted) - 1):
                intron_start = exons_sorted[i][1] + 1
                intron_end = exons_sorted[i + 1][0] - 1
                if intron_end >= intron_start:
                    introns.append({
                        'seqname': info['seqname'], 'start': intron_start,
                        'end': intron_end, 'strand': info['strand'],
                        'gene_name': info['gene_name'], 'feature_type': 'intron',
                        'transcript_id': transcript_id
                    })

        logger.info(f"  CDS: {len(features['CDS']):,}, intron: {len(introns):,}")
        logger.info(f"  tRNA: {len(features['tRNA']):,}, rRNA: {len(features['rRNA']):,}")
        logger.info(f"  snRNA: {len(features['snRNA']):,}, snoRNA: {len(features['snoRNA']):,}")
        logger.info(f"  lncRNA_exon: {len(features['lncRNA_exon']):,}")

        return features, introns

    def load_genome(self, fasta_path):
        """加载基因组序列"""
        logger.info(f"加载基因组: {fasta_path}")
        genome = {}
        for record in SeqIO.parse(fasta_path, "fasta"):
            genome[record.id] = str(record.seq).upper()
        logger.info(f"  加载了 {len(genome)} 条序列")
        return genome

    def extract_sequence(self, genome, seqname, start, end, strand):
        """提取序列（考虑链方向）"""
        if seqname not in genome:
            return None
        seq = genome[seqname][start-1:end]
        if strand == '-':
            seq = str(Seq(seq).reverse_complement())
        return seq

    def process_species(self, species_key, config):
        """处理单个物种"""
        logger.info(f"\n{'='*60}")
        logger.info(f"处理物种: {config['name']} ({config['common_name']})")
        logger.info(f"{'='*60}")

        features, introns = self.parse_gff(config['gff'], config['accession'])
        genome = self.load_genome(config['fasta'])

        samples = []
        failed = 0

        for feature_type, feature_list in features.items():
            for feat in feature_list:
                seq = self.extract_sequence(
                    genome, feat['seqname'], feat['start'], feat['end'], feat['strand']
                )
                if seq:
                    sample = {
                        'sequence': seq, 'label_name': feature_type,
                        'sequence_length': len(seq), 'gene_name': feat['gene_name'],
                        'species': config['accession'], 'seqname': feat['seqname'],
                        'start': feat['start'], 'end': feat['end'],
                        'strand': feat['strand'], 'feature_type': feature_type
                    }
                    if 'transcript_id' in feat:
                        sample['transcript_id'] = feat['transcript_id']
                    samples.append(sample)
                else:
                    failed += 1

        for intron in introns:
            seq = self.extract_sequence(
                genome, intron['seqname'], intron['start'], intron['end'], intron['strand']
            )
            if seq:
                samples.append({
                    'sequence': seq, 'label_name': 'intron',
                    'sequence_length': len(seq), 'gene_name': intron['gene_name'],
                    'species': config['accession'], 'seqname': intron['seqname'],
                    'start': intron['start'], 'end': intron['end'],
                    'strand': intron['strand'], 'feature_type': 'intron',
                    'transcript_id': intron['transcript_id']
                })
            else:
                failed += 1

        if failed > 0:
            logger.warning(f"  ⚠ {failed:,} 条序列提取失败")
        logger.info(f"✓ 提取了 {len(samples):,} 条样本")

        return samples


def main():
    """主函数：提取所有泛化物种数据，分物种保存"""
    logger.info("=" * 70)
    logger.info("泛化物种数据提取 - 开始")
    logger.info("=" * 70)

    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extractor = GeneralizationDataExtractor()

    all_stats = {}

    for species_key, config in SPECIES_CONFIGS.items():
        if not config['gff'].exists() or not config['fasta'].exists():
            logger.error(f"文件不存在: {species_key}")
            continue

        samples = extractor.process_species(species_key, config)

        # 保存单物种数据
        output_file = DATA_OUTPUT_DIR / f"{species_key}_raw.json"
        with open(output_file, 'w') as f:
            json.dump(samples, f, ensure_ascii=False)
        logger.info(f"✓ 保存: {output_file}")

        # 统计
        label_counts = Counter([s['label_name'] for s in samples])
        all_stats[species_key] = {
            'total': len(samples),
            'labels': dict(label_counts)
        }

    # 打印汇总
    logger.info(f"\n{'='*70}")
    logger.info("汇总统计")
    logger.info(f"{'='*70}")
    for sp, stats in all_stats.items():
        logger.info(f"\n{sp}: {stats['total']:,} 样本")
        for label, count in sorted(stats['labels'].items()):
            logger.info(f"  {label}: {count:,}")

    logger.info("\n" + "=" * 70)
    logger.info("泛化物种数据提取 - 完成")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
