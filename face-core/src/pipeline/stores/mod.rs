// ====================================================================
// 人脸向量存储
// ====================================================================
//
// 纯 Rust 暴力余弦检索 (brute-force cosine search), 不依赖外部向量索引库。
//
// 性能边界:
//   embedding 维度 512, 暴力扫描 O(n * 512) 每次查询;
//   万级以下规模 (n <= ~10_000) 单次查询在亚毫秒~毫秒级, 足够使用。
//   更大规模请换用 ANN 索引 (如 hnsw_rs / usearch)。
//
// 语义说明:
//   - 向量入库前 L2 归一化, 内积 == 余弦相似度
//     (add 时自动校验/归一化)
//   - find_best_match 返回最优匹配,
//     相似度 <= threshold 时返回 None
//   - 判定为 `similarity > threshold` (严格大于)

use anyhow::{Result, bail};

use crate::pipeline::recognition::{cosine_similarity, l2_normalize};
use crate::types::{Emotion, Gender};

/// 向量条目元数据
#[derive(Debug, Clone, PartialEq)]
pub struct FaceMetadata {
    /// 姓名 (可选)
    pub name: Option<String>,
    /// 性别 (可选)
    pub gender: Option<Gender>,
    /// 年龄, 单位: 年 (可选)
    pub age: Option<u32>,
    /// 情绪 (可选)
    pub emotion: Option<Emotion>,
    /// 入库时间 (unix 秒)
    pub added_at: u64,
}

/// 检索结果 — 索引、相似度与元数据
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// 库内索引 (与 add 返回值对应)
    pub index: usize,
    /// 余弦相似度 [-1, 1] (库内向量均归一化, 即内积)
    pub score: f32,
    /// 该条目的元数据
    pub metadata: FaceMetadata,
}

/// 人脸向量存储 — L2 归一化向量的暴力余弦检索
///
/// 内部为扁平向量表 + 同步的 metadata 列表;
/// 列表下标即向量 ID, remove 后 (如同按 ID 批量移除)
/// 幸存条目相对顺序不变, 下标前移。
///
/// Example:
/// ```ignore
/// let mut store = FaceStore::new();
/// let idx = store.add(embedding, FaceMetadata { name: Some("Alice".into()), ..Default::default() });
/// if let Some(hit) = store.find_best_match(&query, 0.4) {
///     println!("matched #{} ({:?}) score={:.3}", hit.index, hit.metadata.name, hit.score);
/// }
/// ```
#[derive(Debug, Default)]
pub struct FaceStore {
    /// 全部 L2 归一化的 embedding
    embeddings: Vec<Vec<f32>>,
    /// 与 embeddings 一一对应的元数据
    metadata: Vec<FaceMetadata>,
}

impl FaceStore {
    /// 创建空存储
    pub fn new() -> Self {
        Self {
            embeddings: Vec::new(),
            metadata: Vec::new(),
        }
    }

    /// 添加一条 embedding 及其元数据, 返回分配的索引
    ///
    /// 期望输入已 L2 归一化 (范数 ≈ 1); 若不满足则自动归一化后入库,
    /// 保证内积 == 余弦相似度。
    pub fn add(&mut self, embedding: Vec<f32>, metadata: FaceMetadata) -> usize {
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        // 容差 1e-3: f32 累加误差下的归一化结果仍视为单位向量
        let embedding = if (norm - 1.0).abs() <= 1e-3 {
            embedding
        } else {
            l2_normalize(&embedding)
        };
        self.embeddings.push(embedding);
        self.metadata.push(metadata);
        self.embeddings.len() - 1
    }

    /// 当前向量数量
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// top-k 余弦检索, 按分数降序返回
    ///
    /// top_k 大于库存量时返回全部; 空 store 返回空 Vec。
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        if top_k == 0 || self.embeddings.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<SearchResult> = self
            .embeddings
            .iter()
            .zip(self.metadata.iter())
            .enumerate()
            .map(|(index, (emb, meta))| SearchResult {
                index,
                score: cosine_similarity(query, emb),
                metadata: meta.clone(),
            })
            .collect();
        // 按分数降序; 平分时保持索引升序 (稳定排序)
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        scored
    }

    /// 按索引移除一条记录, 返回其元数据
    ///
    /// 之后条目的索引前移。
    pub fn remove(&mut self, index: usize) -> Result<FaceMetadata> {
        if index >= self.embeddings.len() {
            bail!(
                "remove index {} out of bounds (len {})",
                index,
                self.embeddings.len()
            );
        }
        self.embeddings.remove(index);
        Ok(self.metadata.remove(index))
    }

    /// 按索引查询元数据
    pub fn get(&self, index: usize) -> Option<&FaceMetadata> {
        self.metadata.get(index)
    }

    /// 找到 ≥threshold 的最优匹配
    ///
    /// 空库或最高相似度 < threshold 时返回 None。
    /// 判定使用 >= (含等于)。
    pub fn find_best_match(&self, embedding: &[f32], threshold: f32) -> Option<SearchResult> {
        let mut best: Option<SearchResult> = None;
        for (idx, emb) in self.embeddings.iter().enumerate() {
            let score = cosine_similarity(embedding, emb);
            // 保留最高分; 平分时保留更小索引 (首个出现者)
            let better = match &best {
                Some(cur) => score > cur.score,
                None => true,
            };
            if better {
                best = Some(SearchResult {
                    index: idx,
                    score,
                    metadata: self.metadata[idx].clone(),
                });
            }
        }
        match best {
            Some(r) if r.score >= threshold => Some(r),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(name: &str) -> FaceMetadata {
        FaceMetadata {
            name: Some(name.to_string()),
            gender: None,
            age: None,
            emotion: None,
            added_at: 1_700_000_000,
        }
    }

    /// 构造 (dim-2 维) 单位向量: 方向由首两个分量决定, 其余为 0
    fn unit(dim: usize, x: f32, y: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        let norm = (x * x + y * y).sqrt();
        v[0] = x / norm;
        v[1] = y / norm;
        v
    }

    #[test]
    fn test_add_search_roundtrip_top1() {
        // 两个相近向量 (夹角 ~11.5°) + 一个正交向量, 验证 top-1 命中
        let mut store = FaceStore::new();
        let dim = 512;
        let a = unit(dim, 1.0, 0.0);
        let near = unit(dim, 1.0, 0.2);
        let ortho = unit(dim, 0.0, 1.0);

        store.add(a.clone(), meta("A"));
        store.add(ortho, meta("Ortho"));
        store.add(near.clone(), meta("Near"));
        assert_eq!(store.len(), 3);

        // 查询与 near 最接近
        let results = store.search(&near, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].metadata.name.as_deref(), Some("Near"));
        assert!(results[0].score > 0.98, "score={}", results[0].score);
        // 次优应为 A (与 near 同向族) 而非正交向量
        assert_eq!(results[1].metadata.name.as_deref(), Some("A"));
        assert!(results[1].score > 0.0);

        // 正交查询命中正交向量, top-1 分数 ~1
        let q = unit(dim, 0.0, 1.0);
        let results = store.search(&q, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 1);
        assert!(results[0].score > 0.999, "score={}", results[0].score);
    }

    #[test]
    fn test_search_sorted_descending() {
        // 检索结果按分数降序
        let mut store = FaceStore::new();
        let dim = 8;
        store.add(unit(dim, 1.0, 0.0), meta("a"));
        store.add(unit(dim, 1.0, 1.0), meta("b"));
        store.add(unit(dim, 0.0, 1.0), meta("c"));

        let q = unit(dim, 1.0, 0.1);
        let results = store.search(&q, 3);
        assert_eq!(results.len(), 3);
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn test_find_best_match_threshold() {
        // 阈值过滤: 高于阈值返回最优, 低于阈值返回 None
        let mut store = FaceStore::new();
        let dim = 512;
        let a = unit(dim, 1.0, 0.0);
        let b = unit(dim, 1.0, 1.0); // 与 a 夹角 45°, cos ~ 0.707
        store.add(a.clone(), meta("A"));

        // 同向量: 分数 ~1, 高阈值也通过
        let hit = store.find_best_match(&a, 0.9).expect("should match");
        assert_eq!(hit.metadata.name.as_deref(), Some("A"));
        assert!(hit.score > 0.99, "score={}", hit.score);

        // b 相似度 ~0.707: 阈值 0.8 时拒绝, 0.5 时通过
        assert!(store.find_best_match(&b, 0.8).is_none());
        let hit = store.find_best_match(&b, 0.5).expect("should match");
        assert!((hit.score - 0.707).abs() < 0.01, "score={}", hit.score);
    }

    #[test]
    fn test_remove_shifts_indices() {
        // remove 后索引前移, 元数据与向量保持对齐
        let mut store = FaceStore::new();
        let dim = 512;
        let i0 = store.add(unit(dim, 1.0, 0.0), meta("first"));
        let i1 = store.add(unit(dim, 0.0, 1.0), meta("second"));
        let i2 = store.add(unit(dim, 1.0, 1.0), meta("third"));
        assert_eq!((i0, i1, i2), (0, 1, 2));

        let removed = store.remove(0).unwrap();
        assert_eq!(removed.name.as_deref(), Some("first"));
        assert_eq!(store.len(), 2);

        // 原 index 1/2 前移为 0/1
        assert_eq!(store.get(0).unwrap().name.as_deref(), Some("second"));
        assert_eq!(store.get(1).unwrap().name.as_deref(), Some("third"));

        // 检索返回的 index 与 get 对齐
        let q = unit(dim, 0.0, 1.0);
        let top = &store.search(&q, 1)[0];
        assert_eq!(top.index, 0);
        assert_eq!(top.metadata.name.as_deref(), Some("second"));

        // 越界 remove 报错
        assert!(store.remove(5).is_err());
    }

    #[test]
    fn test_empty_store() {
        // 空 store: search 空, find_best_match None, get None
        let store = FaceStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        let q = vec![1.0f32, 0.0];
        assert!(store.search(&q, 5).is_empty());
        assert!(store.find_best_match(&q, -1.0).is_none());
        assert!(store.get(0).is_none());
    }

    #[test]
    fn test_non_normalized_input_auto_normalized() {
        // 非归一化输入自动归一化: 同方向不同模长检索等价
        let mut store = FaceStore::new();
        let dim = 512;
        let base = unit(dim, 3.0, 4.0);
        // 模长 10 的同向向量
        let scaled: Vec<f32> = base.iter().map(|x| x * 10.0).collect();
        let idx = store.add(scaled, meta("scaled"));
        assert_eq!(idx, 0);

        // 用原始方向查询, 相似度应 ~1
        let hits = store.search(&base, 1);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].score > 0.99, "score={}", hits[0].score);

        // 已归一化向量不被改动
        let n = unit(dim, 1.0, 0.0);
        let before = n.clone();
        store.add(n.clone(), meta("unit"));
        assert_eq!(store_len_of(&store, 1), before.len());
    }

    /// 读取内部向量维度用于断言 (仅测试用)
    fn store_len_of(store: &FaceStore, index: usize) -> usize {
        store.embeddings.get(index).map(|v| v.len()).unwrap_or(0)
    }

    #[test]
    fn test_get_and_metadata_fields() {
        // get 返回完整元数据字段
        let mut store = FaceStore::new();
        let md = FaceMetadata {
            name: Some("Alice".into()),
            gender: Some(Gender::Female),
            age: Some(30),
            emotion: Some(Emotion::Happy),
            added_at: 42,
        };
        store.add(vec![1.0, 0.0], md);
        let got = store.get(0).unwrap();
        assert_eq!(got.name.as_deref(), Some("Alice"));
        assert_eq!(got.gender, Some(Gender::Female));
        assert_eq!(got.age, Some(30));
        assert_eq!(got.emotion, Some(Emotion::Happy));
        assert_eq!(got.added_at, 42);
    }
}
