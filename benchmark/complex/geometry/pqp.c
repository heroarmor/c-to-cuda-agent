/* pqp.c -- proximity queries on triangle meshes using a bounding volume tree
 *
 * Builds two triangulated torus meshes, constructs an AABB tree for each, then
 * performs three PQP-style queries: closest distance, collision at zero
 * tolerance, and "closer than tolerance".
 *
 * Pattern: hierarchical geometry traversal with irregular tree-pair work,
 * branchy pruning, and exact triangle-triangle distance tests.
 * GPU conversion: build or upload the BVHs on the host, traverse many node
 * pairs on the GPU, and preserve the final minimum-distance reduction.
 */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct { double x, y, z; } V3;
typedef struct { V3 a, b, c; int id; } Tri;
typedef struct { V3 lo, hi; int left, right, start, count; } Node;
typedef struct { Node *nodes; int nnode, cap; Tri *tris; } BVH;
typedef struct { int a, b; } Pair;
typedef struct { Pair *p; int n, cap; } Stack;

static V3 v3(double x, double y, double z) { return (V3){ x, y, z }; }
static V3 add(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static V3 sub(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static V3 mul(V3 a, double s) { return v3(a.x * s, a.y * s, a.z * s); }
static double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross(V3 a, V3 b) {
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static double norm2(V3 a) { return dot(a, a); }
static double min3(double a, double b, double c) { return fmin(a, fmin(b, c)); }
static V3 tri_centroid(const Tri *t) { return mul(add(add(t->a, t->b), t->c), 1.0 / 3.0); }

static void expand_point(V3 *lo, V3 *hi, V3 p) {
    lo->x = fmin(lo->x, p.x); lo->y = fmin(lo->y, p.y); lo->z = fmin(lo->z, p.z);
    hi->x = fmax(hi->x, p.x); hi->y = fmax(hi->y, p.y); hi->z = fmax(hi->z, p.z);
}

static int sort_axis = 0;
static int cmp_centroid(const void *pa, const void *pb) {
    V3 ca = tri_centroid((const Tri *)pa);
    V3 cb = tri_centroid((const Tri *)pb);
    double da = (sort_axis == 0) ? ca.x : (sort_axis == 1) ? ca.y : ca.z;
    double db = (sort_axis == 0) ? cb.x : (sort_axis == 1) ? cb.y : cb.z;
    return (da > db) - (da < db);
}

static int build_node(BVH *b, int start, int count) {
    int id = b->nnode++;
    if (b->nnode > b->cap) {
        fprintf(stderr, "internal BVH capacity exceeded\n");
        exit(1);
    }
    Node *n = &b->nodes[id];
    n->lo = v3(DBL_MAX, DBL_MAX, DBL_MAX);
    n->hi = v3(-DBL_MAX, -DBL_MAX, -DBL_MAX);
    n->left = n->right = -1;
    n->start = start;
    n->count = count;
    for (int i = start; i < start + count; ++i) {
        expand_point(&n->lo, &n->hi, b->tris[i].a);
        expand_point(&n->lo, &n->hi, b->tris[i].b);
        expand_point(&n->lo, &n->hi, b->tris[i].c);
    }
    if (count <= 4) return id;

    V3 ext = sub(n->hi, n->lo);
    sort_axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z) ? 1 : 2;
    qsort(b->tris + start, (size_t)count, sizeof(Tri), cmp_centroid);
    int mid = count / 2;
    n->left = build_node(b, start, mid);
    n->right = build_node(b, start + mid, count - mid);
    return id;
}

static BVH make_bvh(Tri *tris, int ntri) {
    BVH b;
    b.tris = tris;
    b.cap = 2 * ntri + 1;
    b.nodes = (Node *)calloc((size_t)b.cap, sizeof(Node));
    if (!b.nodes) { fprintf(stderr, "alloc failed\n"); exit(1); }
    b.nnode = 0;
    build_node(&b, 0, ntri);
    return b;
}

static double bbox_dist2(const Node *a, const Node *b) {
    double dx = 0.0, dy = 0.0, dz = 0.0;
    if (a->hi.x < b->lo.x) dx = b->lo.x - a->hi.x; else if (b->hi.x < a->lo.x) dx = a->lo.x - b->hi.x;
    if (a->hi.y < b->lo.y) dy = b->lo.y - a->hi.y; else if (b->hi.y < a->lo.y) dy = a->lo.y - b->hi.y;
    if (a->hi.z < b->lo.z) dz = b->lo.z - a->hi.z; else if (b->hi.z < a->lo.z) dz = a->lo.z - b->hi.z;
    return dx * dx + dy * dy + dz * dz;
}

static double point_tri_dist2(V3 p, V3 a, V3 b, V3 c) {
    V3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    double d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return norm2(ap);

    V3 bp = sub(p, b);
    double d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return norm2(bp);

    double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        double v = d1 / (d1 - d3);
        return norm2(sub(p, add(a, mul(ab, v))));
    }

    V3 cp = sub(p, c);
    double d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return norm2(cp);

    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        double w = d2 / (d2 - d6);
        return norm2(sub(p, add(a, mul(ac, w))));
    }

    double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        V3 bc = sub(c, b);
        double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return norm2(sub(p, add(b, mul(bc, w))));
    }

    V3 n = cross(ab, ac);
    double nn = norm2(n);
    if (nn == 0.0) return min3(norm2(sub(p, a)), norm2(sub(p, b)), norm2(sub(p, c)));
    double dist = dot(ap, n);
    return (dist * dist) / nn;
}

static double seg_seg_dist2(V3 p1, V3 q1, V3 p2, V3 q2) {
    const double eps = 1e-15;
    V3 d1 = sub(q1, p1), d2 = sub(q2, p2), r = sub(p1, p2);
    double a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
    double s, t;
    if (a <= eps && e <= eps) return norm2(sub(p1, p2));
    if (a <= eps) {
        s = 0.0;
        t = fmax(0.0, fmin(1.0, f / e));
    } else {
        double c = dot(d1, r);
        if (e <= eps) {
            t = 0.0;
            s = fmax(0.0, fmin(1.0, -c / a));
        } else {
            double b = dot(d1, d2);
            double denom = a * e - b * b;
            s = (denom != 0.0) ? fmax(0.0, fmin(1.0, (b * f - c * e) / denom)) : 0.0;
            t = (b * s + f) / e;
            if (t < 0.0) { t = 0.0; s = fmax(0.0, fmin(1.0, -c / a)); }
            else if (t > 1.0) { t = 1.0; s = fmax(0.0, fmin(1.0, (b - c) / a)); }
        }
    }
    return norm2(sub(add(p1, mul(d1, s)), add(p2, mul(d2, t))));
}

static double tri_tri_dist2(const Tri *a, const Tri *b) {
    double d = point_tri_dist2(a->a, b->a, b->b, b->c);
    d = fmin(d, point_tri_dist2(a->b, b->a, b->b, b->c));
    d = fmin(d, point_tri_dist2(a->c, b->a, b->b, b->c));
    d = fmin(d, point_tri_dist2(b->a, a->a, a->b, a->c));
    d = fmin(d, point_tri_dist2(b->b, a->a, a->b, a->c));
    d = fmin(d, point_tri_dist2(b->c, a->a, a->b, a->c));
    V3 ae[3][2] = { { a->a, a->b }, { a->b, a->c }, { a->c, a->a } };
    V3 be[3][2] = { { b->a, b->b }, { b->b, b->c }, { b->c, b->a } };
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            d = fmin(d, seg_seg_dist2(ae[i][0], ae[i][1], be[j][0], be[j][1]));
    return d;
}

static void push(Stack *s, int a, int b) {
    if (s->n == s->cap) {
        s->cap = s->cap ? 2 * s->cap : 1024;
        s->p = (Pair *)realloc(s->p, (size_t)s->cap * sizeof(Pair));
        if (!s->p) { fprintf(stderr, "alloc failed\n"); exit(1); }
    }
    s->p[s->n++] = (Pair){ a, b };
}

static double query_distance(const BVH *a, const BVH *b, int *best_a, int *best_b, long long *node_pairs, long long *tri_pairs) {
    double best = DBL_MAX;
    Stack st = { NULL, 0, 0 };
    push(&st, 0, 0);
    *node_pairs = *tri_pairs = 0;
    *best_a = *best_b = -1;
    while (st.n) {
        Pair p = st.p[--st.n];
        const Node *na = &a->nodes[p.a], *nb = &b->nodes[p.b];
        ++*node_pairs;
        if (bbox_dist2(na, nb) >= best) continue;
        if (na->left < 0 && nb->left < 0) {
            for (int i = na->start; i < na->start + na->count; ++i)
                for (int j = nb->start; j < nb->start + nb->count; ++j) {
                    double d = tri_tri_dist2(&a->tris[i], &b->tris[j]);
                    ++*tri_pairs;
                    if (d < best || (d == best && a->tris[i].id < *best_a)) {
                        best = d;
                        *best_a = a->tris[i].id;
                        *best_b = b->tris[j].id;
                    }
                }
        } else if (nb->left < 0 || (na->left >= 0 && na->count >= nb->count)) {
            push(&st, na->left, p.b);
            push(&st, na->right, p.b);
        } else {
            push(&st, p.a, nb->left);
            push(&st, p.a, nb->right);
        }
    }
    free(st.p);
    return best;
}

static int query_tolerance(const BVH *a, const BVH *b, double tol, long long *node_pairs, long long *tri_pairs) {
    double tol2 = tol * tol;
    Stack st = { NULL, 0, 0 };
    push(&st, 0, 0);
    *node_pairs = *tri_pairs = 0;
    while (st.n) {
        Pair p = st.p[--st.n];
        const Node *na = &a->nodes[p.a], *nb = &b->nodes[p.b];
        ++*node_pairs;
        if (bbox_dist2(na, nb) > tol2) continue;
        if (na->left < 0 && nb->left < 0) {
            for (int i = na->start; i < na->start + na->count; ++i)
                for (int j = nb->start; j < nb->start + nb->count; ++j) {
                    ++*tri_pairs;
                    if (tri_tri_dist2(&a->tris[i], &b->tris[j]) <= tol2) {
                        free(st.p);
                        return 1;
                    }
                }
        } else if (nb->left < 0 || (na->left >= 0 && na->count >= nb->count)) {
            push(&st, na->left, p.b);
            push(&st, na->right, p.b);
        } else {
            push(&st, p.a, nb->left);
            push(&st, p.a, nb->right);
        }
    }
    free(st.p);
    return 0;
}

static V3 transform(V3 p, double angle, double sep) {
    double c = cos(angle), s = sin(angle);
    return v3(c * p.x - s * p.y + sep, s * p.x + c * p.y + 0.11, p.z + 0.07);
}

static int make_torus(Tri *tris, int rings, int tubes, int model, double sep) {
    const double pi = acos(-1.0);
    int nt = 0;
    for (int i = 0; i < rings; ++i) {
        double u0 = 2.0 * pi * i / rings, u1 = 2.0 * pi * (i + 1) / rings;
        for (int j = 0; j < tubes; ++j) {
            double v0 = 2.0 * pi * j / tubes, v1 = 2.0 * pi * (j + 1) / tubes;
            double us[4] = { u0, u1, u1, u0 };
            double vs[4] = { v0, v0, v1, v1 };
            V3 p[4];
            for (int k = 0; k < 4; ++k) {
                double ripple = 1.0 + 0.10 * sin(3.0 * us[k] + 2.0 * vs[k]);
                double major = 1.0, minor = 0.24 * ripple;
                p[k] = v3((major + minor * cos(vs[k])) * cos(us[k]),
                          (major + minor * cos(vs[k])) * sin(us[k]),
                          minor * sin(vs[k]));
                if (model == 1) p[k] = transform(p[k], 0.37, sep);
            }
            tris[nt] = (Tri){ p[0], p[1], p[2], nt }; ++nt;
            tris[nt] = (Tri){ p[0], p[2], p[3], nt }; ++nt;
        }
    }
    return nt;
}

int main(int argc, char **argv) {
    int rings = (argc > 1) ? atoi(argv[1]) : 48;
    int tubes = (argc > 2) ? atoi(argv[2]) : 24;
    double sep = (argc > 3) ? atof(argv[3]) : 0.72;
    double tol = (argc > 4) ? atof(argv[4]) : 0.20;
    if (rings < 8 || tubes < 6 || sep < 0.0 || tol < 0.0) {
        fprintf(stderr, "usage: %s [rings>=8] [tubes>=6] [separation>=0] [tolerance>=0]\n", argv[0]);
        return 2;
    }

    int ntri = 2 * rings * tubes;
    Tri *a = (Tri *)malloc((size_t)ntri * sizeof(Tri));
    Tri *b = (Tri *)malloc((size_t)ntri * sizeof(Tri));
    if (!a || !b) { fprintf(stderr, "alloc failed\n"); return 1; }
    make_torus(a, rings, tubes, 0, sep);
    make_torus(b, rings, tubes, 1, sep);

    clock_t t0 = clock();
    BVH ba = make_bvh(a, ntri);
    BVH bb = make_bvh(b, ntri);
    int ia, ib;
    long long dist_nodes, dist_tris, tol_nodes, tol_tris, col_nodes, col_tris;
    double d2 = query_distance(&ba, &bb, &ia, &ib, &dist_nodes, &dist_tris);
    int colliding = query_tolerance(&ba, &bb, 1e-9, &col_nodes, &col_tris);
    int closer = query_tolerance(&ba, &bb, tol, &tol_nodes, &tol_tris);
    clock_t t1 = clock();

    long long qdist = llround(sqrt(d2) * 1000000000.0);
    long long checksum = qdist ^ ((long long)(ia + 1) * 1000003LL) ^ ((long long)(ib + 1) * 9176LL);
    checksum ^= (long long)colliding * 65537LL ^ (long long)closer * 4099LL;

    printf("pqp rings=%d tubes=%d tris=%d distance=%.9f pair=%d,%d colliding=%d closer_than_tol=%d checksum=%lld (visits=%lld/%lld tol_visits=%lld/%lld collision_visits=%lld/%lld) time=%.3f s\n",
           rings, tubes, ntri, sqrt(d2), ia, ib, colliding, closer, checksum,
           dist_nodes, dist_tris, tol_nodes, tol_tris, col_nodes, col_tris,
           (double)(t1 - t0) / CLOCKS_PER_SEC);

    free(ba.nodes);
    free(bb.nodes);
    free(a);
    free(b);
    return 0;
}
