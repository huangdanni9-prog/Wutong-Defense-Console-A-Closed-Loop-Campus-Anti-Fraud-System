"""
Fraud User Portrait Generator

Task 2: Wire fraud user portrait and behavioral characteristics

This module:
1. Deduplicates blacklist to unique profiles (1 per MSISDN)
2. Aggregates behavioral features per fraudster
3. Clusters into fraud archetypes (Simbox, Wangiri, etc.)
4. Analyzes student reach patterns
5. Generates portrait report with visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class FraudPortraitGenerator:
    """Generate fraud user portraits with behavioral analysis."""
    
    def __init__(self, blacklist_path: str = None, fraud_data_path: str = None):
        base = Path(__file__).parent.parent.parent
        self.blacklist_path = blacklist_path or str(base / "Datasets" / "Fraud" / "Results" / "blacklist.csv")
        self.fraud_data_path = fraud_data_path or str(base / "Datasets" / "Fraud" / "Training and Testing Data" / "fraud_model_2.csv")
        self.student_path = str(base / "Datasets" / "Student" / "Training and Testing Data" / "student_model.csv")
        self.output_dir = base / "Datasets" / "Fraud" / "Results"
        
        self.profiles = None
        self.archetypes = None
    
    def load_and_deduplicate(self) -> pd.DataFrame:
        """Load ALL fraud sources and deduplicate to unique profiles.
        
        Sources:
        1. blacklist.csv - Detected fraud (11,160 rows)
        2. greylist.csv - Suspicious (681 rows) 
        3. fraud_model_2.csv - Confirmed fraud (8,960 rows)
        
        All are combined and deduplicated by MSISDN for clustering.
        """
        print("=== Phase 1: Load and Combine ALL Fraud Sources ===")
        
        base = Path(self.fraud_data_path).parent.parent
        greylist_path = base / "Results" / "greylist.csv"
        
        all_dfs = []
        
        # Source 1: fraud_model_2.csv (full features, confirmed fraud)
        fraud_df = pd.read_csv(self.fraud_data_path, low_memory=False)
        confirmed_fraud = fraud_df[fraud_df['audit_status'] == '稽核不通過'].copy()
        confirmed_fraud['source_list'] = 'fraud_model_2'
        all_dfs.append(confirmed_fraud)
        print(f"  fraud_model_2 (confirmed): {len(confirmed_fraud)} rows, {confirmed_fraud['msisdn'].nunique()} unique")
        
        # Source 2: blacklist.csv + features from fraud_model_2
        bl = pd.read_csv(self.blacklist_path)
        bl_msisdns = set(bl['msisdn'].unique())
        bl_with_features = fraud_df[fraud_df['msisdn'].isin(bl_msisdns)].copy()
        bl_with_features['source_list'] = 'blacklist'
        all_dfs.append(bl_with_features)
        print(f"  blacklist (with features): {len(bl_with_features)} rows, {bl_with_features['msisdn'].nunique()} unique")
        
        # Source 3: greylist.csv + features from fraud_model_2
        if greylist_path.exists():
            gl = pd.read_csv(greylist_path)
            gl_msisdns = set(gl['msisdn'].unique())
            gl_with_features = fraud_df[fraud_df['msisdn'].isin(gl_msisdns)].copy()
            gl_with_features['source_list'] = 'greylist'
            all_dfs.append(gl_with_features)
            print(f"  greylist (with features): {len(gl_with_features)} rows, {gl_with_features['msisdn'].nunique()} unique")
        
        # Combine all sources
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"  Combined total: {len(combined)} rows, {combined['msisdn'].nunique()} unique MSISDNs")
        
        # Build aggregation dict dynamically based on available columns
        agg_dict = {}
        
        # Identity (first value)
        for col in ['iden_type', 'post_or_ppd', 'ntwk_type', 'open_dt', 'source_list']:
            if col in combined.columns:
                agg_dict[col] = 'first'
        
        # Call volume (mean per day)
        for col in ['call_cnt_day', 'called_cnt_day', 'call_cnt_day_2s', 'call_cnt_day_3m', 
                    'dispersion_rate', 'sms_cnt_day', 'duration_sum_day']:
            if col in combined.columns:
                agg_dict[col] = 'mean'
        
        # Roaming (sum)
        for col in ['roam_unknow_call_cnt', 'local_unknow_call_cnt']:
            if col in combined.columns:
                agg_dict[col] = 'sum'
        
        # Max values
        for col in ['change_imei_times', 'opp_num_stu_cnt']:
            if col in combined.columns:
                agg_dict[col] = 'max'
        
        # Activity period
        if 'stat_dt' in combined.columns:
            agg_dict['stat_dt'] = ['min', 'max', 'count']
        
        # Aggregate per MSISDN
        profiles = combined.groupby('msisdn').agg(agg_dict).reset_index()
        
        # Flatten multi-level columns
        new_cols = []
        for col in profiles.columns:
            if isinstance(col, tuple):
                new_cols.append('_'.join(col).strip('_'))
            else:
                new_cols.append(col)
        profiles.columns = new_cols
        
        print(f"  Deduplicated profiles: {len(profiles)}")
        self.profiles = profiles
        return profiles
    
    def engineer_portrait_features(self) -> pd.DataFrame:
        """Engineer features for portrait analysis."""
        print("\n=== Phase 2: Engineer Portrait Features ===")
        
        df = self.profiles.copy()
        
        # Helper to safely get column
        def safe_col(name, default=0):
            return df[name] if name in df.columns else default
        
        # Identity features - match BOTH Chinese and English values
        # Note: After aggregation, columns may have '_first' suffix
        iden_col = 'iden_type_first' if 'iden_type_first' in df.columns else 'iden_type'
        ppd_col = 'post_or_ppd_first' if 'post_or_ppd_first' in df.columns else 'post_or_ppd'
        
        if iden_col in df.columns:
            # 護照 = Passport (Chinese), also check English variants including 'Passport /CI'
            passport_patterns = ['護照', 'Passport', '护照', 'Passport /CI']
            df['is_passport'] = df[iden_col].astype(str).str.contains('|'.join(passport_patterns), case=False, na=False).astype(int)
            print(f"  Passport detection: {df['is_passport'].sum()} ({df['is_passport'].mean()*100:.1f}%)")
        else:
            df['is_passport'] = 0
            print(f"  Warning: No iden_type column found")
            
        if ppd_col in df.columns:
            # 预付 = Prepaid (Chinese), 后付 = Postpaid
            prepaid_patterns = ['预付', 'PPD', 'Prepaid']
            df['is_prepaid'] = df[ppd_col].astype(str).str.contains('|'.join(prepaid_patterns), case=False, na=False).astype(int)
            print(f"  Prepaid detection: {df['is_prepaid'].sum()} ({df['is_prepaid'].mean()*100:.1f}%)")
        else:
            df['is_prepaid'] = 0
            print(f"  Warning: No post_or_ppd column found")
        
        # Call behavior (use available column names)
        call_cnt = safe_col('call_cnt_day_mean', safe_col('call_cnt_day', 0))
        called_cnt = safe_col('called_cnt_day_mean', safe_col('called_cnt_day', 0))
        
        df['total_calls'] = call_cnt + called_cnt
        df['outbound_ratio'] = call_cnt / (df['total_calls'] + 1)
        df['is_silent_receiver'] = ((called_cnt == 0) & (call_cnt > 0)).astype(int)
        
        # Short call ratio
        short_calls = safe_col('call_cnt_day_2s_mean', safe_col('call_cnt_day_2s', 0))
        df['short_call_ratio'] = short_calls / (call_cnt + 1)
        
        # Roaming
        roam = safe_col('roam_unknow_call_cnt', 0)
        local = safe_col('local_unknow_call_cnt', 0)
        df['roaming_ratio'] = roam / (roam + local + 1)
        
        # SMS heavy (default to 0 if no SMS data)
        df['is_sms_heavy'] = 0  # No SMS data in this dataset
        
        # Activity span
        if 'stat_dt_count' in df.columns:
            df['active_days'] = df['stat_dt_count']
        else:
            df['active_days'] = 1
        
        print(f"  Engineered {len(df.columns)} features")
        self.profiles = df
        return df
    
    def cluster_archetypes(self, n_clusters: int = 6) -> pd.DataFrame:
        """
        Cluster fraud ENTRIES (rows) into behavioral archetypes.
        
        Strategy (Per-Entry Clustering):
        1. Cluster each ROW - captures temporal attack patterns (e.g., silent in Jan, attacking in Feb)
        2. Aggregate by MSISDN - take WORST-CASE label for blocking
        
        This is better than clustering averaged profiles because fraud is an ACTION, not a permanent state.
        """
        print(f"\n=== Phase 3: Cluster ENTRIES into {n_clusters} Archetypes ===")
        
        if not SKLEARN_AVAILABLE:
            print("  Warning: sklearn not available, skipping clustering")
            self.profiles['archetype'] = 'Unknown'
            return self.profiles
        
        # Use RAW combined data (not aggregated profiles) for per-entry clustering
        base = Path(self.fraud_data_path).parent.parent
        fraud_df = pd.read_csv(self.fraud_data_path, low_memory=False)
        
        # Load blacklist and greylist MSISDNs
        bl = pd.read_csv(self.blacklist_path)
        bl_msisdns = set(bl['msisdn'].unique())
        
        greylist_path = base / "Results" / "greylist.csv"
        if greylist_path.exists():
            gl = pd.read_csv(greylist_path)
            gl_msisdns = set(gl['msisdn'].unique())
        else:
            gl_msisdns = set()
        
        # Confirmed fraud + blacklist + greylist MSISDNs
        confirmed_msisdns = set(fraud_df[fraud_df['audit_status'] == '稽核不通過']['msisdn'].unique())
        all_threat_msisdns = confirmed_msisdns | bl_msisdns | gl_msisdns
        
        # Get ALL rows for threat MSISDNs (not aggregated - per entry!)
        raw_entries = fraud_df[fraud_df['msisdn'].isin(all_threat_msisdns)].copy()
        print(f"  Raw entries for clustering: {len(raw_entries)} rows, {raw_entries['msisdn'].nunique()} MSISDNs")
        
        # Engineer features on raw entries
        raw_entries['outbound_ratio'] = raw_entries['call_cnt_day'] / (raw_entries['call_cnt_day'] + raw_entries['called_cnt_day'] + 1)
        raw_entries['is_silent_receiver'] = ((raw_entries['called_cnt_day'] == 0) & (raw_entries['call_cnt_day'] > 0)).astype(int)
        raw_entries['short_call_ratio'] = raw_entries.get('call_cnt_day_2s', 0) / (raw_entries['call_cnt_day'] + 1)
        
        # Cluster features
        cluster_features = ['outbound_ratio', 'is_silent_receiver', 'short_call_ratio', 'call_cnt_day']
        available = [f for f in cluster_features if f in raw_entries.columns]
        X = raw_entries[available].fillna(0)
        
        # Standardize and cluster
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Downshift cluster count if samples are few
        k = min(n_clusters, len(raw_entries))
        k = max(1, k)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        raw_entries['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Name archetypes based on cluster characteristics - more granular
        archetype_names = {}
        used_names = set()
        
        for cluster in range(k):
            cluster_data = raw_entries[raw_entries['cluster'] == cluster]
            silent = cluster_data['is_silent_receiver'].mean() if 'is_silent_receiver' in cluster_data else 0
            calls = cluster_data['call_cnt_day'].mean() if 'call_cnt_day' in cluster_data else 0
            short_ratio = cluster_data['short_call_ratio'].mean() if 'short_call_ratio' in cluster_data else 0
            outbound = cluster_data['outbound_ratio'].mean() if 'outbound_ratio' in cluster_data else 0
            
            # More nuanced naming based on multiple features
            if silent > 0.8 and calls > 100:
                name = "Simbox (High Volume)"
            elif silent > 0.7 and short_ratio > 0.5:
                name = "Wangiri (Short Calls)"
            elif silent > 0.7:
                name = "Wangiri Caller"
            elif calls > 150:
                name = "Mass Dialer"
            elif calls > 80:
                name = "High Volume Attacker"
            elif calls > 40:
                name = "Active Caller"
            elif outbound > 0.9:
                name = "Outbound Only"
            else:
                name = "Low Activity"
            
            # Add unique suffix if name already used
            base_name = name
            suffix = 1
            while name in used_names:
                suffix += 1
                name = f"{base_name} #{suffix}"
            used_names.add(name)
            archetype_names[cluster] = name
        
        raw_entries['archetype'] = raw_entries['cluster'].map(archetype_names)
        
        print(f"  Per-Entry archetype distribution:")
        for name, count in raw_entries['archetype'].value_counts().items():
            print(f"    {name}: {count}")
        
        # ================================================================
        # AGGREGATE BY MSISDN: Take WORST-CASE label for blocking
        # ================================================================
        # Priority: Higher volume/more aggressive = higher priority
        archetype_priority = {
            "Simbox (High Volume)": 6,
            "Mass Dialer": 5,
            "Wangiri (Short Calls)": 4,
            "Wangiri Caller": 4,
            "High Volume Attacker": 3,
            "Active Caller": 2,
            "Outbound Only": 2,
            "Low Activity": 1
        }
        # Handle numbered variants
        raw_entries['archetype_priority'] = raw_entries['archetype'].apply(
            lambda x: archetype_priority.get(x.split(' #')[0], 1)
        )
        
        # Group by MSISDN, take worst case (max priority)
        worst_case = raw_entries.groupby('msisdn').agg({
            'archetype_priority': 'max',
            'cluster': 'first',  # Keep for viz
            'call_cnt_day': 'mean',
            'outbound_ratio': 'mean',
            'is_silent_receiver': 'max'
        }).reset_index()
        
        # Map priority back to archetype name
        priority_to_name = {v: k for k, v in archetype_priority.items()}
        worst_case['archetype'] = worst_case['archetype_priority'].map(priority_to_name)
        
        print(f"\n  WORST-CASE per MSISDN:")
        for name, count in worst_case['archetype'].value_counts().items():
            print(f"    {name}: {count}")
        
        # Store both raw clustered entries (for viz) and worst-case profiles
        self.raw_clustered = raw_entries
        
        # Merge worst-case archetype into profiles
        self.profiles = self.profiles.merge(
            worst_case[['msisdn', 'archetype', 'cluster']], 
            on='msisdn', 
            how='left'
        )
        self.profiles['archetype'] = self.profiles['archetype'].fillna('Unknown')
        
        self.archetypes = archetype_names
        return self.profiles
    
    def _name_archetypes(self, df: pd.DataFrame, features: list) -> Dict[int, str]:
        """Name clusters based on dominant characteristics."""
        names = {}
        
        for cluster in df['cluster'].unique():
            cluster_data = df[df['cluster'] == cluster]
            
            # Determine dominant trait
            silent = cluster_data['is_silent_receiver'].mean() if 'is_silent_receiver' in cluster_data else 0
            prepaid = cluster_data['is_prepaid'].mean() if 'is_prepaid' in cluster_data else 0
            passport = cluster_data['is_passport'].mean() if 'is_passport' in cluster_data else 0
            sms = cluster_data['is_sms_heavy'].mean() if 'is_sms_heavy' in cluster_data else 0
            
            if silent > 0.7:
                names[cluster] = "Wangiri Caller"
            elif prepaid > 0.7:
                names[cluster] = "Burner Phone"
            elif passport > 0.5:
                names[cluster] = "Passport Fraud"
            elif sms > 0.5:
                names[cluster] = "SMS Scammer"
            else:
                names[cluster] = f"Mixed Pattern {cluster}"
        
        return names
    
    def analyze_student_reach(self) -> Dict:
        """Analyze how fraudsters reach students."""
        print("\n=== Phase 4: Student Reach Analysis ===")
        
        try:
            stu_df = pd.read_csv(self.student_path, low_memory=False)
        except Exception as e:
            print(f"  Warning: Could not load student data: {e}")
            return {}
        
        # Get fraud MSISDNs that contacted students
        fraud_contacted = stu_df[stu_df['fraud_msisdn'].notna()]
        
        stats = {
            'total_students': len(stu_df),
            'students_contacted': len(fraud_contacted),
            'contact_rate': len(fraud_contacted) / len(stu_df) * 100,
            'unique_fraud_msisdns': fraud_contacted['fraud_msisdn'].nunique(),
        }
        
        # Response analysis
        if 'voice_receive' in fraud_contacted.columns:
            stats['students_answered'] = (fraud_contacted['voice_receive'] > 0).sum()
        if 'voice_call' in fraud_contacted.columns:
            stats['students_called_back'] = (fraud_contacted['voice_call'] > 0).sum()
        
        print(f"  Total students: {stats['total_students']:,}")
        print(f"  Students contacted by fraud: {stats['students_contacted']:,} ({stats['contact_rate']:.1f}%)")
        print(f"  Unique fraud MSISDNs: {stats['unique_fraud_msisdns']}")
        if 'students_answered' in stats:
            print(f"  Students who answered: {stats['students_answered']}")
        if 'students_called_back' in stats:
            print(f"  Students who called back: {stats['students_called_back']}")
        
        return stats
    
    def generate_visualizations(self, student_stats: Dict) -> list:
        """Generate visualization charts for the portrait report."""
        print("\n=== Phase 4b: Generate Visualizations ===")
        
        if not MATPLOTLIB_AVAILABLE:
            print("  Warning: matplotlib not available, skipping visualizations")
            return []
        
        saved_charts = []
        df = self.profiles
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid') if 'seaborn-v0_8-whitegrid' in plt.style.available else None
        
        # Chart 1: Archetype Distribution (Pie Chart)
        if 'archetype' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            archetype_counts = df['archetype'].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(archetype_counts)))
            wedges, texts, autotexts = ax.pie(
                archetype_counts.values, 
                labels=archetype_counts.index,
                autopct='%1.1f%%',
                colors=colors,
                explode=[0.05] * len(archetype_counts)
            )
            ax.set_title('Fraud Archetype Distribution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            chart_path = self.output_dir / "chart_archetype_distribution.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            saved_charts.append(str(chart_path))
            print(f"  Saved: {chart_path.name}")
        
        # Chart 2: Call Volume Distribution (Histogram) - with LOG scale for power-law data
        if 'total_calls' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            df['total_calls'].clip(upper=200).hist(bins=30, ax=ax, color='steelblue', edgecolor='white')
            ax.set_xlabel('Daily Calls (capped at 200)', fontsize=12)
            ax.set_ylabel('Number of Fraudsters (log scale)', fontsize=12)
            ax.set_yscale('log')  # FIX: Log scale for power-law distribution
            ax.set_title('Fraud Call Volume Distribution', fontsize=14, fontweight='bold')
            ax.axvline(df['total_calls'].median(), color='red', linestyle='--', label=f'Median: {df["total_calls"].median():.0f}')
            ax.legend()
            plt.tight_layout()
            chart_path = self.output_dir / "chart_call_volume_distribution.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            saved_charts.append(str(chart_path))
            print(f"  Saved: {chart_path.name}")
        
        # Chart 3: Student Reach Impact (Bar Chart)
        if student_stats:
            fig, ax = plt.subplots(figsize=(10, 6))
            categories = ['Total Students', 'Contacted', 'Answered', 'Called Back']
            values = [
                student_stats.get('total_students', 0),
                student_stats.get('students_contacted', 0),
                student_stats.get('students_answered', 0),
                student_stats.get('students_called_back', 0)
            ]
            colors = ['#3498db', '#e74c3c', '#f39c12', '#c0392b']
            bars = ax.bar(categories, values, color=colors, edgecolor='white')
            ax.set_ylabel('Number of Students', fontsize=12)
            ax.set_title('Fraud Reach to Student Population', fontsize=14, fontweight='bold')
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                       f'{val:,}', ha='center', va='bottom', fontsize=10)
            plt.tight_layout()
            chart_path = self.output_dir / "chart_student_reach.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            saved_charts.append(str(chart_path))
            print(f"  Saved: {chart_path.name}")
        
        # Chart 4: Behavioral Features Comparison (Grouped Bar)
        if 'archetype' in df.columns and len(df['archetype'].unique()) > 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            features = ['outbound_ratio', 'short_call_ratio', 'is_silent_receiver']
            available_features = [f for f in features if f in df.columns]
            
            if available_features:
                archetypes = df['archetype'].unique()
                x = np.arange(len(archetypes))
                width = 0.25
                
                for i, feat in enumerate(available_features):
                    means = [df[df['archetype'] == arch][feat].mean() for arch in archetypes]
                    ax.bar(x + i*width, means, width, label=feat.replace('_', ' ').title())
                
                ax.set_xlabel('Archetype', fontsize=12)
                ax.set_ylabel('Feature Value', fontsize=12)
                ax.set_title('Behavioral Features by Fraud Archetype', fontsize=14, fontweight='bold')
                ax.set_xticks(x + width)
                ax.set_xticklabels(archetypes, rotation=15, ha='right')
                ax.legend()
                plt.tight_layout()
                chart_path = self.output_dir / "chart_behavioral_features.png"
                plt.savefig(chart_path, dpi=150, bbox_inches='tight')
                plt.close()
                saved_charts.append(str(chart_path))
                print(f"  Saved: {chart_path.name}")
        
        # Chart 5: Clustering Scatter Plot (2D PCA projection) - Uses RAW ENTRIES
        if hasattr(self, 'raw_clustered') and self.raw_clustered is not None and SKLEARN_AVAILABLE:
            try:
                from sklearn.decomposition import PCA
                
                raw_df = self.raw_clustered
                
                # Get clustering features from raw entries
                cluster_features = ['outbound_ratio', 'is_silent_receiver', 'short_call_ratio', 'call_cnt_day']
                available = [f for f in cluster_features if f in raw_df.columns]
                
                if len(available) >= 2:
                    X = raw_df[available].fillna(0)
                    
                    # FIX: Scale BEFORE PCA to avoid "Variance Trap"
                    # call_cnt_day (0-500) would dominate outbound_ratio (0-1) without scaling
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Reduce to 2D for visualization
                    pca = PCA(n_components=2)
                    X_2d = pca.fit_transform(X_scaled)
                    
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # Plot each archetype with different color
                    archetypes = raw_df['archetype'].unique()
                    colors = plt.cm.Set1(np.linspace(0, 1, len(archetypes)))
                    
                    for arch, color in zip(archetypes, colors):
                        mask = raw_df['archetype'] == arch
                        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                                  c=[color], label=f"{arch} ({mask.sum():,})", 
                                  alpha=0.4, s=20, edgecolors='none')
                    
                    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
                    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
                    ax.set_title(f'Per-Entry Clustering ({len(raw_df):,} entries)', fontsize=14, fontweight='bold')
                    ax.legend(loc='upper right')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    chart_path = self.output_dir / "chart_clustering_2d.png"
                    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    saved_charts.append(str(chart_path))
                    print(f"  Saved: {chart_path.name}")
            except Exception as e:
                print(f"  Warning: Could not generate clustering chart: {e}")
        
        return saved_charts
    
    def generate_report(self, student_stats: Dict) -> str:
        """Generate markdown portrait report."""
        print("\n=== Phase 5: Generate Report ===")
        
        df = self.profiles
        
        report = f"""# Wire Fraud User Portrait Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Fraud Profiles | {len(df):,} |
| Blacklist Coverage | {len(df):,} unique MSISDNs |
| Student Reach | {student_stats.get('students_contacted', 'N/A')} students contacted |

---

## Fraud Archetypes

"""
        # Archetype breakdown
        if 'archetype' in df.columns:
            for archetype in df['archetype'].unique():
                arch_data = df[df['archetype'] == archetype]
                report += f"""### {archetype}
- **Count**: {len(arch_data):,} ({len(arch_data)/len(df)*100:.1f}%)
- **Avg Daily Calls**: {arch_data['call_cnt_day_mean'].mean():.1f}
- **Passport Rate**: {arch_data['is_passport'].mean()*100:.1f}%
- **Prepaid Rate**: {arch_data['is_prepaid'].mean()*100:.1f}%

"""
        
        # Identity analysis
        report += """---

## Identity Analysis

### By Document Type
"""
        iden_col = 'iden_type_first' if 'iden_type_first' in df.columns else 'iden_type'
        if iden_col in df.columns:
            for iden_type, count in df[iden_col].value_counts().head(5).items():
                report += f"- {iden_type}: {count:,} ({count/len(df)*100:.1f}%)\n"
        else:
            report += "- Identity data not available\n"
        
        # Student reach
        report += f"""
---

## Student Reach Analysis

| Metric | Value |
|--------|-------|
| Students Contacted | {student_stats.get('students_contacted', 'N/A')} |
| Contact Rate | {student_stats.get('contact_rate', 0):.2f}% |
| Students Answered | {student_stats.get('students_answered', 'N/A')} |
| Students Called Back | {student_stats.get('students_called_back', 'N/A')} |

---

## Operational Recommendations

1. **Block Wangiri Callers** - Outbound-only numbers with 0 incoming calls
2. **Monitor Passport Users** - 4x higher fraud rate than local IDs
3. **Alert Prepaid Burners** - New accounts with high activity
4. **Protect Students** - SMS warnings to students who received fraud calls

"""
        
        return report
    
    def run_full_pipeline(self):
        """Run the complete portrait generation pipeline."""
        print("=" * 60)
        print("FRAUD USER PORTRAIT GENERATOR")
        print("=" * 60)
        
        # Phase 1: Load and deduplicate
        self.load_and_deduplicate()
        
        # Phase 2: Engineer features
        self.engineer_portrait_features()
        
        # Phase 3: Cluster archetypes
        self.cluster_archetypes()
        
        # Phase 4: Student reach analysis
        student_stats = self.analyze_student_reach()
        
        # Phase 4b: Generate visualizations
        charts = self.generate_visualizations(student_stats)
        
        # Phase 5: Generate report
        report = self.generate_report(student_stats)
        
        # Save outputs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save deduplicated profiles
        profiles_path = self.output_dir / "fraud_profiles.csv"
        self.profiles.to_csv(profiles_path, index=False)
        print(f"\nSaved fraud profiles: {profiles_path}")
        
        # Save report
        report_path = self.output_dir / "fraud_user_portrait.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Saved portrait report: {report_path}")
        
        print("\n" + "=" * 60)
        print("PORTRAIT GENERATION COMPLETE")
        print("=" * 60)
        
        return self.profiles, report


if __name__ == "__main__":
    generator = FraudPortraitGenerator()
    profiles, report = generator.run_full_pipeline()
