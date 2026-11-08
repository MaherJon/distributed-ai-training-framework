package mahe.ui

import androidx.compose.material3.darkColorScheme
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.unit.sp

/**
 * 550W 风格主题：深色底、细线网格、单色琥珀强调 + 等宽字体。
 * 不依赖"大量蓝色"堆砌科技感——以排版、结构线、扫描动效为主。
 */
object MaheTheme {
    val bg = Color(0xFF0A0B0D)
    val panel = Color(0xFF101216)
    val gridLine = Color(0xFF1E222A)
    val accent = Color(0xFFE8A33D)      // 琥珀
    val accentDim = Color(0xFF8A6420)
    val textPrimary = Color(0xFFD8DCE3)
    val textDim = Color(0xFF6B7280)
    val ok = Color(0xFF7FB069)
    val warn = Color(0xFFE8A33D)
    val dead = Color(0xFF8C3B3B)

    val colors = darkColorScheme(
        background = bg,
        surface = panel,
        primary = accent,
        onBackground = textPrimary,
        onSurface = textPrimary,
    )

    val mono = TextStyle(
        fontFamily = FontFamily.Monospace,
        fontSize = 12.sp,
        color = textPrimary,
        lineHeight = 16.sp,
    )
    val monoDim = mono.copy(color = textDim)
    val title = mono.copy(fontSize = 14.sp, color = accent, letterSpacing = 2.sp)
    val bigNumber = mono.copy(fontSize = 28.sp, color = textPrimary)
}
