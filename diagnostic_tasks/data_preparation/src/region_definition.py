"""
区域定义模块 - 从GFF注释计算各类基因组区域

区域定义:
- Gene body: GFF中gene特征的[start, end]区间
- Gene zone: Gene body两端各扩展2kb
- Ambiguous zone: 距离gene body 2-10kb的区域（不参与训练）
- Deep intergenic: 距离gene body ≥10kb的区域
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GENE_ZONE_EXTEND, AMBIGUOUS_BUFFER_MIN, AMBIGUOUS_BUFFER_MAX,
    DEEP_INTERGENIC_MIN_DIST, BLOCK_SIZE
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GenomicRegion:
    """基因组区域数据类"""
    seqname: str        # 染色体/scaffold名称
    start: int          # 起始位置 (1-based)
    end: int            # 结束位置 (1-based, inclusive)
    strand: str = '+'   # 链方向
    region_type: str = ''  # 区域类型
    gene_id: str = ''   # 关联的基因ID (如果有)
    gene_name: str = '' # 基因名称
    attributes: Dict = None  # 其他属性

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def overlaps(self, other: 'GenomicRegion') -> bool:
        """检查是否与另一个区域重叠"""
        if self.seqname != other.seqname:
            return False
        return not (self.end < other.start or self.start > other.end)

    def distance_to(self, other: 'GenomicRegion') -> int:
        """计算到另一个区域的距离（不重叠时）"""
        if self.seqname != other.seqname:
            return float('inf')
        if self.overlaps(other):
            return 0
        if self.end < other.start:
            return other.start - self.end
        return self.start - other.end


class RegionDefinition:
    """基因组区域定义类"""

    def __init__(
        self,
        gene_zone_extend: int = GENE_ZONE_EXTEND,
        ambiguous_min: int = AMBIGUOUS_BUFFER_MIN,
        ambiguous_max: int = AMBIGUOUS_BUFFER_MAX,
        deep_intergenic_min: int = DEEP_INTERGENIC_MIN_DIST,
        block_size: int = BLOCK_SIZE
    ):
        self.gene_zone_extend = gene_zone_extend
        self.ambiguous_min = ambiguous_min
        self.ambiguous_max = ambiguous_max
        self.deep_intergenic_min = deep_intergenic_min
        self.block_size = block_size

        # 存储各类区域
        self.genes: Dict[str, List[GenomicRegion]] = defaultdict(list)  # 按染色体存储
        self.gene_zones: Dict[str, List[GenomicRegion]] = defaultdict(list)
        self.deep_intergenic: Dict[str, List[GenomicRegion]] = defaultdict(list)
        self.chromosome_lengths: Dict[str, int] = {}

        # 基因内部结构
        self.cds_regions: Dict[str, List[GenomicRegion]] = defaultdict(list)
        self.intron_regions: Dict[str, List[GenomicRegion]] = defaultdict(list)
        self.utr_regions: Dict[str, List[GenomicRegion]] = defaultdict(list)
        self.ncrna_regions: Dict[str, List[GenomicRegion]] = defaultdict(list)

        # ncRNA 子类型
        self.ncrna_subtypes: Dict[str, Dict[str, List[GenomicRegion]]] = defaultdict(lambda: defaultdict(list))

    def parse_gff(self, gff_path: Path, fasta_path: Optional[Path] = None) -> None:
        """解析GFF文件，提取基因和各类特征"""
        logger.info(f"解析GFF文件: {gff_path}")

        # 如果提供了FASTA，先获取染色体长度
        if fasta_path:
            self._parse_fasta_lengths(fasta_path)

        # 存储转录本信息用于计算intron
        transcripts: Dict[str, List[GenomicRegion]] = defaultdict(list)  # transcript_id -> exons
        transcript_parents: Dict[str, str] = {}  # transcript_id -> gene_id

        gene_count = 0
        feature_counts = defaultdict(int)

        with open(gff_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) < 9:
                    continue

                seqname, source, feature_type, start, end, score, strand, frame, attributes_str = parts
                start, end = int(start), int(end)

                # 解析属性
                attributes = self._parse_attributes(attributes_str)

                # 处理不同特征类型
                if feature_type == 'gene':
                    gene_id = attributes.get('ID', attributes.get('gene_id', f'gene_{gene_count}'))
                    gene_name = attributes.get('Name', attributes.get('gene', gene_id))
                    gene_biotype = attributes.get('gene_biotype', '')

                    region = GenomicRegion(
                        seqname=seqname, start=start, end=end, strand=strand,
                        region_type='gene', gene_id=gene_id, gene_name=gene_name,
                        attributes={'gene_biotype': gene_biotype}
                    )
                    self.genes[seqname].append(region)
                    gene_count += 1
                    feature_counts['gene'] += 1

                elif feature_type == 'CDS':
                    gene_id = self._get_gene_id_from_attributes(attributes)
                    gene_name = attributes.get('gene', attributes.get('Name', ''))
                    region = GenomicRegion(
                        seqname=seqname, start=start, end=end, strand=strand,
                        region_type='CDS', gene_id=gene_id, gene_name=gene_name,
                        attributes=attributes
                    )
                    self.cds_regions[seqname].append(region)
                    feature_counts['CDS'] += 1

                elif feature_type == 'exon':
                    # 用于计算intron
                    parent = attributes.get('Parent', '')
                    if parent:
                        region = GenomicRegion(
                            seqname=seqname, start=start, end=end, strand=strand,
                            region_type='exon', attributes=attributes
                        )
                        transcripts[parent].append(region)
                    feature_counts['exon'] += 1

                elif feature_type in ('five_prime_UTR', "5'UTR", 'three_prime_UTR', "3'UTR"):
                    gene_id = self._get_gene_id_from_attributes(attributes)
                    region = GenomicRegion(
                        seqname=seqname, start=start, end=end, strand=strand,
                        region_type='UTR', gene_id=gene_id, attributes=attributes
                    )
                    self.utr_regions[seqname].append(region)
                    feature_counts['UTR'] += 1

                elif feature_type in ('mRNA', 'transcript', 'lnc_RNA', 'ncRNA', 'primary_transcript'):
                    # 记录转录本的父基因
                    transcript_id = attributes.get('ID', '')
                    parent_gene = attributes.get('Parent', '')
                    if transcript_id and parent_gene:
                        transcript_parents[transcript_id] = parent_gene

                elif feature_type in ('tRNA', 'rRNA', 'snRNA', 'snoRNA'):
                    gene_id = self._get_gene_id_from_attributes(attributes)
                    gene_name = attributes.get('gene', attributes.get('Name', ''))
                    region = GenomicRegion(
                        seqname=seqname, start=start, end=end, strand=strand,
                        region_type=feature_type, gene_id=gene_id, gene_name=gene_name,
                        attributes=attributes
                    )
                    self.ncrna_regions[seqname].append(region)
                    self.ncrna_subtypes[seqname][feature_type].append(region)
                    feature_counts[feature_type] += 1

                elif feature_type == 'lnc_RNA':
                    gene_id = self._get_gene_id_from_attributes(attributes)
                    gene_name = attributes.get('gene', attributes.get('Name', ''))
                    region = GenomicRegion(
                        seqname=seqname, start=start, end=end, strand=strand,
                        region_type='lncRNA', gene_id=gene_id, gene_name=gene_name,
                        attributes=attributes
                    )
                    self.ncrna_regions[seqname].append(region)
                    self.ncrna_subtypes[seqname]['lncRNA'].append(region)
                    feature_counts['lncRNA'] += 1

        # 计算intron区域
        self._compute_introns(transcripts, transcript_parents)

        # 对每个染色体的基因按位置排序
        for seqname in self.genes:
            self.genes[seqname].sort(key=lambda x: x.start)

        logger.info(f"✓ 解析完成:")
        logger.info(f"  染色体数: {len(self.genes)}")
        for feat, count in sorted(feature_counts.items()):
            logger.info(f"  {feat}: {count:,}")

    def _parse_fasta_lengths(self, fasta_path: Path) -> None:
        """从FASTA文件获取染色体长度"""
        logger.info(f"解析FASTA获取染色体长度: {fasta_path}")
        current_seqname = None
        current_length = 0

        with open(fasta_path, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    if current_seqname:
                        self.chromosome_lengths[current_seqname] = current_length
                    # 提取序列名（第一个空格前的部分）
                    current_seqname = line[1:].split()[0]
                    current_length = 0
                else:
                    current_length += len(line.strip())

            if current_seqname:
                self.chromosome_lengths[current_seqname] = current_length

        logger.info(f"✓ 获取 {len(self.chromosome_lengths)} 个染色体长度")

    def _parse_attributes(self, attributes_str: str) -> Dict[str, str]:
        """解析GFF属性字段"""
        attributes = {}
        for item in attributes_str.split(';'):
            item = item.strip()
            if '=' in item:
                key, value = item.split('=', 1)
                attributes[key] = value
        return attributes

    def _get_gene_id_from_attributes(self, attributes: Dict) -> str:
        """从属性中提取基因ID"""
        # 尝试多种可能的属性名
        for key in ['gene', 'gene_id', 'Parent', 'locus_tag']:
            if key in attributes:
                value = attributes[key]
                # 如果是Parent，可能需要进一步处理
                if key == 'Parent' and ',' in value:
                    value = value.split(',')[0]
                return value
        return ''

    def _compute_introns(self, transcripts: Dict[str, List[GenomicRegion]],
                         transcript_parents: Dict[str, str]) -> None:
        """从exon计算intron区域"""
        logger.info("计算intron区域...")
        intron_count = 0

        for transcript_id, exons in transcripts.items():
            if len(exons) < 2:
                continue

            # 按位置排序exon
            exons.sort(key=lambda x: x.start)
            seqname = exons[0].seqname
            strand = exons[0].strand
            gene_id = transcript_parents.get(transcript_id, '')

            # 计算相邻exon之间的intron
            for i in range(len(exons) - 1):
                intron_start = exons[i].end + 1
                intron_end = exons[i + 1].start - 1

                if intron_end >= intron_start:  # 确保intron长度 > 0
                    region = GenomicRegion(
                        seqname=seqname, start=intron_start, end=intron_end,
                        strand=strand, region_type='intron', gene_id=gene_id
                    )
                    self.intron_regions[seqname].append(region)
                    intron_count += 1

        logger.info(f"✓ 计算得到 {intron_count:,} 个intron区域")

    def compute_gene_zones(self) -> None:
        """计算Gene zone (gene body ± extend)"""
        logger.info(f"计算Gene zone (gene body ± {self.gene_zone_extend}bp)...")

        for seqname, genes in self.genes.items():
            chrom_length = self.chromosome_lengths.get(seqname, float('inf'))

            for gene in genes:
                zone_start = max(1, gene.start - self.gene_zone_extend)
                zone_end = min(chrom_length, gene.end + self.gene_zone_extend)

                zone = GenomicRegion(
                    seqname=seqname, start=zone_start, end=zone_end,
                    strand=gene.strand, region_type='gene_zone',
                    gene_id=gene.gene_id, gene_name=gene.gene_name
                )
                self.gene_zones[seqname].append(zone)

        total_zones = sum(len(zones) for zones in self.gene_zones.values())
        logger.info(f"✓ 生成 {total_zones:,} 个Gene zone")

    def compute_deep_intergenic(self) -> None:
        """计算Deep intergenic区域 (距离gene body ≥ deep_intergenic_min)"""
        logger.info(f"计算Deep intergenic区域 (距离gene body ≥ {self.deep_intergenic_min}bp)...")

        for seqname, genes in self.genes.items():
            if not genes:
                continue

            chrom_length = self.chromosome_lengths.get(seqname)
            if not chrom_length:
                continue

            # 合并所有gene body区域（处理重叠）
            merged_genes = self._merge_overlapping_regions(genes)

            # 计算每个gene的"禁区"（gene body + ambiguous zone）
            forbidden_regions = []
            for gene in merged_genes:
                forbidden_start = max(1, gene.start - self.deep_intergenic_min + 1)
                forbidden_end = min(chrom_length, gene.end + self.deep_intergenic_min - 1)
                forbidden_regions.append((forbidden_start, forbidden_end))

            # 合并禁区
            forbidden_regions.sort()
            merged_forbidden = []
            for start, end in forbidden_regions:
                if merged_forbidden and start <= merged_forbidden[-1][1] + 1:
                    merged_forbidden[-1] = (merged_forbidden[-1][0], max(merged_forbidden[-1][1], end))
                else:
                    merged_forbidden.append((start, end))

            # 计算deep intergenic（禁区的补集）
            prev_end = 0
            for forbidden_start, forbidden_end in merged_forbidden:
                if forbidden_start > prev_end + 1:
                    region = GenomicRegion(
                        seqname=seqname, start=prev_end + 1, end=forbidden_start - 1,
                        region_type='deep_intergenic'
                    )
                    self.deep_intergenic[seqname].append(region)
                prev_end = forbidden_end

            # 处理最后一段
            if prev_end < chrom_length:
                region = GenomicRegion(
                    seqname=seqname, start=prev_end + 1, end=chrom_length,
                    region_type='deep_intergenic'
                )
                self.deep_intergenic[seqname].append(region)

        total_regions = sum(len(regions) for regions in self.deep_intergenic.values())
        total_length = sum(r.length for regions in self.deep_intergenic.values() for r in regions)
        logger.info(f"✓ 生成 {total_regions:,} 个Deep intergenic区域, 总长度 {total_length:,} bp")

    def _merge_overlapping_regions(self, regions: List[GenomicRegion]) -> List[GenomicRegion]:
        """合并重叠的区域"""
        if not regions:
            return []

        sorted_regions = sorted(regions, key=lambda x: x.start)
        merged = [sorted_regions[0]]

        for region in sorted_regions[1:]:
            if region.start <= merged[-1].end + 1:
                # 重叠或相邻，合并
                merged[-1] = GenomicRegion(
                    seqname=merged[-1].seqname,
                    start=merged[-1].start,
                    end=max(merged[-1].end, region.end),
                    region_type=merged[-1].region_type
                )
            else:
                merged.append(region)

        return merged

    def assign_blocks_to_intergenic(self) -> Dict[str, List[Tuple[GenomicRegion, int]]]:
        """为deep intergenic区域分配block ID

        block_id 的设计原则：
        - 同一染色体的同一个 500kb 基因组区间共享同一个 block_id
        - block_id = (seqname_index * max_blocks_per_chrom) + block_index
        - 这样确保同一局部基因组区域的样本在划分时被分到同一个集合
        """
        logger.info(f"为Deep intergenic分配Block ID (block size: {self.block_size}bp)...")

        block_assignments = defaultdict(list)

        # 为每个染色体分配一个索引，用于生成全局唯一的 block_id
        seqname_list = sorted(self.deep_intergenic.keys())
        seqname_to_idx = {name: idx for idx, name in enumerate(seqname_list)}

        # 估算每个染色体最大的 block 数量（用于生成不重叠的 block_id）
        max_chrom_length = max(self.chromosome_lengths.values()) if self.chromosome_lengths else 1_000_000_000
        max_blocks_per_chrom = (max_chrom_length // self.block_size) + 1

        for seqname in seqname_list:
            seqname_idx = seqname_to_idx[seqname]

            for region in self.deep_intergenic[seqname]:
                # 计算该区域跨越的 block 索引
                start_block_idx = region.start // self.block_size
                end_block_idx = region.end // self.block_size

                for block_idx in range(start_block_idx, end_block_idx + 1):
                    # 计算该 block 在当前 region 中的实际范围
                    block_start = max(region.start, block_idx * self.block_size)
                    block_end = min(region.end, (block_idx + 1) * self.block_size - 1)

                    if block_end >= block_start:
                        sub_region = GenomicRegion(
                            seqname=seqname, start=block_start, end=block_end,
                            region_type='deep_intergenic'
                        )
                        # 全局唯一的 block_id：同一染色体同一 500kb 区间共享
                        global_block_id = seqname_idx * max_blocks_per_chrom + block_idx
                        block_assignments[seqname].append((sub_region, global_block_id))

        total_blocks = len(set(bid for regions in block_assignments.values() for _, bid in regions))
        total_sub_regions = sum(len(regions) for regions in block_assignments.values())
        logger.info(f"✓ 分配了 {total_blocks:,} 个唯一Block, {total_sub_regions:,} 个子区域")

        return block_assignments

    def get_gene_clusters(self) -> Dict[str, List[Set[str]]]:
        """获取基因簇（处理重叠的gene zone）

        返回格式: {seqname: [set(gene_id1, gene_id2, ...), set(...), ...]}
        每个 set 包含一个簇中所有基因的 gene_id
        这个格式与 DataSplitter._merge_gene_clusters() 期待的输入一致
        """
        logger.info("计算基因簇（处理重叠的gene zone）...")

        clusters: Dict[str, List[Set[str]]] = defaultdict(list)

        for seqname, zones in self.gene_zones.items():
            if not zones:
                continue

            # 按起始位置排序
            sorted_zones = sorted(zones, key=lambda x: x.start)

            # 当前簇的基因ID集合
            current_cluster_ids: Set[str] = {sorted_zones[0].gene_id}
            current_cluster_end = sorted_zones[0].end

            for zone in sorted_zones[1:]:
                # 检查是否与当前簇重叠
                if zone.start <= current_cluster_end:
                    # 重叠，加入当前簇
                    current_cluster_ids.add(zone.gene_id)
                    current_cluster_end = max(current_cluster_end, zone.end)
                else:
                    # 不重叠，保存当前簇，开始新簇
                    clusters[seqname].append(current_cluster_ids)
                    current_cluster_ids = {zone.gene_id}
                    current_cluster_end = zone.end

            # 保存最后一个簇
            clusters[seqname].append(current_cluster_ids)

        total_clusters = sum(len(c) for c in clusters.values())
        total_genes = sum(len(gene_set) for c in clusters.values() for gene_set in c)
        logger.info(f"✓ {total_genes:,} 个基因分为 {total_clusters:,} 个簇")

        return clusters

    def get_statistics(self) -> Dict:
        """获取区域统计信息"""
        stats = {
            'chromosomes': len(self.genes),
            'total_genes': sum(len(g) for g in self.genes.values()),
            'total_gene_zones': sum(len(z) for z in self.gene_zones.values()),
            'total_deep_intergenic': sum(len(r) for r in self.deep_intergenic.values()),
            'total_cds': sum(len(c) for c in self.cds_regions.values()),
            'total_introns': sum(len(i) for i in self.intron_regions.values()),
            'total_utrs': sum(len(u) for u in self.utr_regions.values()),
            'total_ncrna': sum(len(n) for n in self.ncrna_regions.values()),
            'ncrna_subtypes': {
                subtype: sum(
                    len(self.ncrna_subtypes[seqname].get(subtype, []))
                    for seqname in self.ncrna_subtypes
                )
                for subtype in ['lncRNA', 'tRNA', 'rRNA', 'snRNA', 'snoRNA']
            },
            'gene_zone_total_length': sum(z.length for zones in self.gene_zones.values() for z in zones),
            'deep_intergenic_total_length': sum(r.length for regions in self.deep_intergenic.values() for r in regions),
        }
        return stats


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import GFF_DIR, FASTA_DIR, SPECIES_CONFIG

    # 使用第一个物种测试
    species_id = list(SPECIES_CONFIG.keys())[0]
    species_info = SPECIES_CONFIG[species_id]

    gff_path = GFF_DIR / species_info['gff']
    fasta_path = FASTA_DIR / species_info['fasta']

    print(f"测试物种: {species_info['name']} ({species_info['common_name']})")
    print(f"GFF: {gff_path}")
    print(f"FASTA: {fasta_path}")

    if gff_path.exists():
        rd = RegionDefinition()
        rd.parse_gff(gff_path, fasta_path if fasta_path.exists() else None)
        rd.compute_gene_zones()
        rd.compute_deep_intergenic()

        stats = rd.get_statistics()
        print("\n统计信息:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v:,}")
            else:
                print(f"  {key}: {value:,}")
    else:
        print(f"GFF文件不存在: {gff_path}")
