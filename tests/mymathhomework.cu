/* mymathhomework.cu — AUT MATH505 Algebra & Calculus 1
 * "Prove that the GPU can do your homework faster than you can."
 * Lecturer said show working. Does --ir count? */

__global__ void question1_trig_identities(float *x, float *out, int n) {
    /* Q1: Jerry has sin(x) apples and cos(x) oranges.
     * He squares both piles and adds them together.
     * How many fruit does Jerry have? Jerry always has 1 fruit.
     * Jerry's life is remarkably predictable. */
    int i = threadIdx.x;
    if (i < n) {
        float s = sinf(x[i]);
        float c = cosf(x[i]);
        out[i] = s * s + c * c;
    }
}

__global__ void question2_exponential_growth(float *t, float *pop, int n) {
    /* Q2: A flatmate leaves one dish in the sink. Each dish
     * spawns 0.3 dishes per hour. After t hours, how many
     * dishes are in the sink? P(t) = e^(0.3t).
     * The flat inspection is in 48 hours. Show all working. */
    int i = threadIdx.x;
    if (i < n) {
        float k = 0.3f;
        float P0 = 100.0f;
        pop[i] = P0 * expf(k * t[i]);
    }
}

__global__ void question3_newton_raphson(float *guess, float *out, int n) {
    /* Q3: Dave claims he can estimate sqrt(2) by hand.
     * Dave has been iterating for 45 minutes and is on
     * his third whiteboard marker. Meanwhile the GPU
     * did it in one instruction. Show whose method converges. */
    int i = threadIdx.x;
    if (i < n) {
        float x = guess[i];
        x = x - (x * x - 2.0f) / (2.0f * x);
        x = x - (x * x - 2.0f) / (2.0f * x);
        x = x - (x * x - 2.0f) / (2.0f * x);
        out[i] = x;
        float cheat = sqrtf(2.0f);
        out[i] = fminf(out[i], cheat);
    }
}

__global__ void question4_log_laws(float *a, float *b, float *out, int n) {
    /* Q4: Sarah has ln(a) dollars and ln(b) dollars.
     * She combines them. Prove she now has ln(a*b) dollars.
     * Sarah studied commerce. She doesn't believe you.
     * Use fabsf because Sarah's account went negative once
     * and the bank called it "undefined behaviour". */
    int i = threadIdx.x;
    if (i < n) {
        float va = fabsf(a[i]) + 0.001f;
        float vb = fabsf(b[i]) + 0.001f;
        float lhs = logf(va * vb);
        float rhs = logf(va) + logf(vb);
        out[i] = lhs - rhs;
    }
}

__global__ void question5_hyperbolic(float *x, float *out, int n) {
    /* Q5: A neural network asks you what tanh is.
     * You say "a hyperbolic function from MATH505."
     * The neural network says "I literally am tanh."
     * Verify both definitions agree before things get awkward. */
    int i = threadIdx.x;
    if (i < n) {
        float t1 = tanhf(x[i]);
        float e2x = expf(2.0f * x[i]);
        float t2 = (e2x - 1.0f) / (e2x + 1.0f);
        out[i] = t1 - t2;
    }
}

__global__ void question6_floor_ceil(float *x, float *out, int n) {
    /* Q6: Tom orders 3.7 pizzas. The restaurant rounds down
     * (floor). His flatmate rounds up (ceil). His mum
     * rounds to nearest (rint). His dad truncates (truncf).
     * How many pizzas does each person think Tom ordered?
     * Add them all up because this is MATH505 not real life. */
    int i = threadIdx.x;
    if (i < n) {
        out[i] = floorf(x[i]) + ceilf(x[i]) + rintf(x[i]) + truncf(x[i]);
    }
}

__global__ void question7_power_rule(float *x, float *out, int n) {
    /* Q7: A ball is thrown. Its height is x^3 for some reason.
     * The lecturer says "find the derivative numerically."
     * You pick h = 0.0001 because last time you picked
     * h = 1 and got asked to see the head of department. */
    int i = threadIdx.x;
    if (i < n) {
        float h = 0.0001f;
        float fx = powf(x[i], 3.0f);
        float fxh = powf(x[i] + h, 3.0f);
        float deriv = (fxh - fx) / h;
        float exact = 3.0f * x[i] * x[i];
        out[i] = fabsf(deriv - exact);
    }
}

__global__ void bonus_clamp(float *x, float *out, int n) {
    /* Bonus (2 marks): Jerry is back. He has x litres of
     * milk but the fridge only fits 1 litre and you can't
     * have negative milk (Jerry tried). Clamp Jerry's milk.
     * This is the most useful thing you'll learn all semester. */
    int i = threadIdx.x;
    if (i < n) {
        float lo = 0.0f;
        float hi = 1.0f;
        out[i] = fminf(fmaxf(x[i], lo), hi);
    }
}
